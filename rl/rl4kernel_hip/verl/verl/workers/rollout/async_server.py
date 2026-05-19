# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import fastapi
import ray
import uvicorn
from ray.exceptions import GetTimeoutError
from omegaconf import DictConfig
from starlette.requests import Request

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class AsyncServerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self, *, force_dedicated_loop: bool = False):
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = threading.Event()
        self.server_ready_metadata: Dict[str, Any] = {}
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._force_dedicated_loop = force_dedicated_loop
        self._loop_mode = "dedicated_thread" if force_dedicated_loop else "auto"

        if force_dedicated_loop:
            self._server_loop = asyncio.new_event_loop()
            self._server_thread = threading.Thread(target=self._run_server_loop, daemon=True)
            self._server_thread.start()
            logger.warning(
                "[async-sidecar-debug] AsyncServerBase using dedicated background loop address=%s force_dedicated_loop=%s",
                self.address,
                force_dedicated_loop,
            )
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._server_loop = asyncio.new_event_loop()
                self._server_thread = threading.Thread(target=self._run_server_loop, daemon=True)
                self._server_thread.start()
                self._loop_mode = "dedicated_thread"
                logger.warning(
                    "[async-sidecar-debug] AsyncServerBase fell back to dedicated background loop address=%s",
                    self.address,
                )
            else:
                self._server_loop = loop
                self._loop_mode = "borrowed_current_loop"
                logger.warning(
                    "[async-sidecar-debug] AsyncServerBase borrowing current loop address=%s",
                    self.address,
                )
                loop.create_task(self._start_fastapi_server())

    def _run_server_loop(self):
        assert self._server_loop is not None
        asyncio.set_event_loop(self._server_loop)
        self._server_loop.run_until_complete(self._start_fastapi_server())

    async def _start_fastapi_server(self):
        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            print(f"FastAPI listen on {self.address}:{self.port}")
            self.server_ready_metadata = {"address": self.address, "port": self.port}
            self.server_ready.set()
            yield

            # There's no way to gracefully restart uvicorn server if port is already in use,
            # so we exit the process directly and let AsyncLLMServerManager restart it.
            print("FastAPI shutdown, maybe address already in use, exit process immediately.")
            os._exit(-1)

        app = fastapi.FastAPI(lifespan=lifespan)
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        await server.serve()

    async def get_server_address(self) -> str:
        """Get FastAPI server address."""
        print(f"[AsyncServerBase] wait server_ready address={self.address} port={self.port}", flush=True)
        await asyncio.to_thread(self.server_ready.wait)
        print(f"[AsyncServerBase] server_ready address={self.address} port={self.port}", flush=True)
        return f"{self.address}:{self.port}"

    def run_coroutine_sync(self, coroutine, timeout_s: float | None = None):
        if self._server_loop is None:
            raise RuntimeError("AsyncServerBase does not have a running event loop.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._server_loop)
        return future.result(timeout=timeout_s)

    def shutdown(self):
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server.force_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)

    @abstractmethod
    async def chat_completion(self, raw_request: Request):
        """OpenAI chat completion API.

        API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        raise NotImplementedError

    @abstractmethod
    async def init_engine(self):
        """Init async LLM engine."""
        raise NotImplementedError

    @abstractmethod
    async def wake_up(self):
        """Wake up engine to load model weights and build kv cache."""
        raise NotImplementedError

    @abstractmethod
    async def sleep(self):
        """Sleep engine to offload model weights and discard kv cache."""
        raise NotImplementedError


@dataclass
class AsyncServerSlot:
    dp_rank: int
    worker_rank: int
    node_id: str
    server_name: str
    worker_names: List[str] = field(default_factory=list)
    ray_address: str | None = None
    spawn_attempts: int = 0
    server_handle: Any | None = None
    server_address: str | None = None
    engine_initialized: bool = False
    last_error: str | None = None
    runtime_info: Dict[str, Any] = field(default_factory=dict)


class _ServerMethodProxy:
    def __init__(self, call_remote):
        self._call_remote = call_remote

    def remote(self, *args, **kwargs):
        return self._call_remote(*args, **kwargs)


class WorkerOwnedAsyncServerProxy:
    def __init__(self, worker_handle, server_name: str):
        self.worker_handle = worker_handle
        self.server_name = server_name
        self.get_server_address = _ServerMethodProxy(lambda: worker_handle.get_colocated_async_server_address.remote(server_name))
        self.init_engine = _ServerMethodProxy(lambda: worker_handle.init_colocated_async_server.remote(server_name))
        self.describe_runtime = _ServerMethodProxy(lambda: worker_handle.describe_colocated_async_server.remote(server_name))
        self.wake_up = _ServerMethodProxy(lambda: worker_handle.wake_up_colocated_async_server.remote(server_name))
        self.sleep = _ServerMethodProxy(lambda: worker_handle.sleep_colocated_async_server.remote(server_name))
        self.shutdown = _ServerMethodProxy(lambda: worker_handle.stop_colocated_async_server.remote(server_name))

    def __repr__(self) -> str:
        return f"WorkerOwnedAsyncServerProxy(server_name={self.server_name!r}, worker_handle={self.worker_handle!r})"


class AsyncLLMServerManager:
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize AsyncLLMServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
        """
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size
        self.max_server_init_retries = max(1, int(os.environ.get("VERL_ASYNC_SERVER_MAX_RETRIES", "3")))
        self.server_address_timeout_s = float(os.environ.get("VERL_ASYNC_SERVER_ADDRESS_TIMEOUT_S", "120"))
        self.server_init_timeout_s = float(os.environ.get("VERL_ASYNC_SERVER_INIT_TIMEOUT_S", "1800"))
        self.server_describe_timeout_s = float(os.environ.get("VERL_ASYNC_SERVER_DESCRIBE_TIMEOUT_S", "60"))
        self._servers_ready = False
        self._ready = False

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size
        ray_address = getattr(ray.get_runtime_context(), "gcs_address", None)

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )

        self.sidecar_slots = [
            AsyncServerSlot(
                dp_rank=rollout_dp_rank,
                worker_rank=rollout_dp_rank * self.rollout_tp_size,
                node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                server_name=self._build_server_name(rollout_dp_rank),
                worker_names=list(
                    self.worker_group.worker_names[
                        rollout_dp_rank * self.rollout_tp_size : (rollout_dp_rank + 1) * self.rollout_tp_size
                    ]
                ),
                ray_address=ray_address,
            )
            for rollout_dp_rank in range(self.rollout_dp_size)
        ]
        for slot in self.sidecar_slots:
            self._initialize_server_slot(slot, server_class)
            self.async_llm_servers[slot.dp_rank] = slot.server_handle
            self.server_addresses[slot.dp_rank] = slot.server_address
        self._servers_ready = True

        # Init user provided chat scheduler in sperate thread.
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_exception: Exception = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(target=self._init_chat_scheduler, daemon=True)
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()
        self._assert_ready(require_chat_scheduler=True)
        self._ready = True

    def _build_server_name(self, rollout_dp_rank: int) -> str:
        prefix = self.worker_group.name_prefix.rstrip("_")
        if prefix:
            return f"{prefix}_async_llm_server_{rollout_dp_rank}"
        return f"async_llm_server_{rollout_dp_rank}"

    def _kill_server_handle(self, server_handle: Any | None) -> None:
        if server_handle is None:
            return
        if isinstance(server_handle, WorkerOwnedAsyncServerProxy):
            try:
                ray.get(server_handle.shutdown.remote())
            except Exception as exc:
                logger.warning("Failed to shutdown worker-owned async sidecar proxy=%s: %s", server_handle, exc)
            return
        try:
            ray.kill(server_handle)
        except Exception as exc:
            logger.warning("Failed to kill async rollout server handle=%s: %s", server_handle, exc)

    def _spawn_server_for_slot(self, slot: AsyncServerSlot, server_class: Type[AsyncServerBase]):
        if self.config.rollout.name == "vllm":
            sharing_worker = self.worker_group.workers[slot.worker_rank]
            ray.get(
                sharing_worker.spawn_colocated_async_server.remote(
                    rollout_backend=self.config.rollout.name,
                    config=self.full_config,
                    dp_size=self.rollout_dp_size,
                    dp_rank=slot.dp_rank,
                    wg_prefix=self.worker_group.name_prefix,
                    server_name=slot.server_name,
                    worker_names=slot.worker_names,
                    ray_address=slot.ray_address,
                    server_runtime_env={
                        "env_vars": {
                            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
                        }
                    },
                )
            )
            return WorkerOwnedAsyncServerProxy(sharing_worker, slot.server_name)
        return server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=slot.node_id,
                soft=False,
            ),
            name=slot.server_name,
        ).remote(self.full_config, self.rollout_dp_size, slot.dp_rank, self.worker_group.name_prefix)

    def _initialize_server_slot(self, slot: AsyncServerSlot, server_class: Type[AsyncServerBase]) -> None:
        last_exception: Exception | None = None
        while slot.spawn_attempts < self.max_server_init_retries:
            slot.spawn_attempts += 1
            slot.engine_initialized = False
            slot.runtime_info = {}
            logger.info(
                "Initializing async rollout server dp_rank=%s worker_rank=%s node_id=%s server_name=%s attempt=%s/%s",
                slot.dp_rank,
                slot.worker_rank,
                slot.node_id,
                slot.server_name,
                slot.spawn_attempts,
                self.max_server_init_retries,
            )
            server = self._spawn_server_for_slot(slot, server_class)
            try:
                logger.warning(
                    "[async-sidecar-debug] waiting address dp_rank=%s server_name=%s timeout_s=%s",
                    slot.dp_rank,
                    slot.server_name,
                    self.server_address_timeout_s,
                )
                server_address = self._ray_get_stage(
                    server.get_server_address.remote(),
                    timeout_s=self.server_address_timeout_s,
                    stage="get_server_address",
                    slot=slot,
                )
                logger.warning(
                    "[async-sidecar-debug] resolved address dp_rank=%s server_name=%s address=%s",
                    slot.dp_rank,
                    slot.server_name,
                    server_address,
                )
                logger.warning(
                    "[async-sidecar-debug] init engine dp_rank=%s server_name=%s timeout_s=%s",
                    slot.dp_rank,
                    slot.server_name,
                    self.server_init_timeout_s,
                )
                self._ray_get_stage(
                    server.init_engine.remote(),
                    timeout_s=self.server_init_timeout_s,
                    stage="init_engine",
                    slot=slot,
                )
                logger.warning(
                    "[async-sidecar-debug] init engine done dp_rank=%s server_name=%s",
                    slot.dp_rank,
                    slot.server_name,
                )
                runtime_info = {}
                if self.config.rollout.name == "vllm":
                    logger.warning(
                        "[async-sidecar-debug] describe runtime dp_rank=%s server_name=%s timeout_s=%s",
                        slot.dp_rank,
                        slot.server_name,
                        self.server_describe_timeout_s,
                    )
                    runtime_info = self._ray_get_stage(
                        server.describe_runtime.remote(),
                        timeout_s=self.server_describe_timeout_s,
                        stage="describe_runtime",
                        slot=slot,
                    )
                    if not runtime_info.get("engine_initialized", False):
                        raise RuntimeError(
                            f"async rollout server {slot.server_name} finished init_engine without engine_initialized"
                        )
                    logger.warning(
                        "[async-sidecar-debug] describe runtime done dp_rank=%s server_name=%s runtime=%s",
                        slot.dp_rank,
                        slot.server_name,
                        runtime_info,
                    )
                slot.server_handle = server
                slot.server_address = server_address
                slot.engine_initialized = True
                slot.last_error = None
                slot.runtime_info = runtime_info
                logger.info(
                    "Async rollout server ready dp_rank=%s server_name=%s address=%s runtime=%s",
                    slot.dp_rank,
                    slot.server_name,
                    slot.server_address,
                    slot.runtime_info,
                )
                return
            except Exception as exc:
                last_exception = exc
                slot.last_error = repr(exc)
                logger.exception(
                    "Async rollout server init failed dp_rank=%s server_name=%s attempt=%s/%s",
                    slot.dp_rank,
                    slot.server_name,
                    slot.spawn_attempts,
                    self.max_server_init_retries,
                )
                self._kill_server_handle(server)
                slot.server_handle = None
                slot.server_address = None
        raise RuntimeError(
            f"Failed to initialize async rollout server dp_rank={slot.dp_rank} "
            f"server_name={slot.server_name} after {self.max_server_init_retries} attempts. "
            f"Last error: {slot.last_error}"
        ) from last_exception

    def _ray_get_stage(self, object_ref, *, timeout_s: float, stage: str, slot: AsyncServerSlot):
        start = time.time()
        try:
            result = ray.get(object_ref, timeout=timeout_s)
        except GetTimeoutError as exc:
            elapsed = time.time() - start
            raise TimeoutError(
                f"Timed out in async rollout stage={stage} dp_rank={slot.dp_rank} "
                f"server_name={slot.server_name} after {elapsed:.2f}s (timeout={timeout_s}s)"
            ) from exc
        elapsed = time.time() - start
        logger.warning(
            "[async-sidecar-debug] stage=%s dp_rank=%s server_name=%s elapsed_s=%.2f",
            stage,
            slot.dp_rank,
            slot.server_name,
            elapsed,
        )
        return result

    def _assert_ready(self, require_chat_scheduler: bool = False) -> None:
        if not self._servers_ready:
            raise RuntimeError("Async rollout servers are not initialized yet.")
        unready_slots = [
            slot.server_name
            for slot in self.sidecar_slots
            if slot.server_handle is None or not slot.engine_initialized or not slot.server_address
        ]
        if unready_slots:
            raise RuntimeError(f"Async rollout servers are not ready: {unready_slots}")
        if not require_chat_scheduler:
            return
        if self.chat_scheduler_exception is not None:
            raise RuntimeError("Chat scheduler failed to initialize.") from self.chat_scheduler_exception
        if self.chat_scheduler is None or self.chat_scheduler_loop is None:
            raise RuntimeError("Chat scheduler is not initialized.")

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)

        try:
            self.chat_scheduler = ChatCompletionScheduler(
                config=self.full_config,
                server_addresses=self.server_addresses,
            )
        except Exception as e:
            logger.exception(f"chat_scheduler init error: {e}")
            self.chat_scheduler_exception = e
        finally:
            self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    def wake_up(self):
        """Wake up all vllm instances."""
        self._assert_ready()
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        self._assert_ready()
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def submit_chat_completions(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Dict[str, Any],
    ):
        """Submit a chat completion request to chat scheduler and wait until it is done.
        To submit multiple requests in parallel, please use `generate_sequences` instead.

        Args: same as ChatCompletionScheduler.submit_chat_completions.
        """
        self._assert_ready(require_chat_scheduler=True)
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler._submit_chat_completions_semaphore(
                messages=messages,
                request_id=None,
                sampling_params=sampling_params,
            ),
            self.chat_scheduler_loop,
        )
        future.result()

    def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        self._assert_ready(require_chat_scheduler=True)

        future = asyncio.run_coroutine_threadsafe(self.chat_scheduler.generate_sequences(prompts, **sampling_params), self.chat_scheduler_loop)
        return future.result()


def async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    """Get async server class.

    Args:
        rollout_backend: str, rollout backend, should be "vllm" or "sglang".

    Returns:
        Type[AsyncServerBase]: async server class.
    """
    if rollout_backend == "vllm":
        from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer

        return AsyncvLLMServer
    elif rollout_backend == "sglang":
        from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSglangServer

        return AsyncSglangServer
    else:
        raise NotImplementedError
