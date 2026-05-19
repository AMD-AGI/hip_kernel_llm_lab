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
import json
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
        worker_names_json = os.environ.get("VERL_VLLM_WORKER_NAMES_JSON", "").strip()
        ray_address = os.environ.get("VERL_VLLM_RAY_ADDRESS", "auto").strip() or "auto"
        explicit_namespace = os.environ.get("VERL_VLLM_RAY_NAMESPACE", "").strip()

        if worker_names_json:
            namespace = explicit_namespace or None
            if not ray.is_initialized():
                if namespace:
                    ray.init(address=ray_address, namespace=namespace)
                else:
                    ray.init(address=ray_address)
            actor_names = json.loads(worker_names_json)
            vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            assert len(actor_names) == vllm_tp_size, (
                f"instance_id: {self.vllm_config.instance_id} received explicit worker_names={actor_names}, "
                f"but tensor_parallel_size={vllm_tp_size}."
            )
            print(
                f"instance_id: {self.vllm_config.instance_id} initializes with explicit worker actors: {actor_names}",
                flush=True,
            )
        else:
            fields = self.vllm_config.instance_id.split(":")
            assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
            namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

            # Make sure subprocess in same namespace as parent actor.
            # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
            if not ray.is_initialized():
                ray.init(address=ray_address, namespace=namespace)
            vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]
            assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
                f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but "
                f"vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected. "
                "The current async vLLM path still depends on legacy WorkerDict actor naming."
            )

            def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
                fields = actor_name.split(":")
                assert len(fields) == 2, f"invalid actor name: {actor_name}"
                pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
                return pg_index, local_rank

            # sort actor names by pg_index and local_rank
            actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
            actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} initializes with external actors: {actor_names}")

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(init_worker) start", flush=True)
        self.collective_rpc("init_worker", args=([kwargs],))
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(init_worker) done", flush=True)
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(init_device) start", flush=True)
        self.collective_rpc("init_device")
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(init_device) done", flush=True)
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(load_model) start", flush=True)
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} collective_rpc(load_model) done", flush=True)
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        print(
            f"instance_id: {self.vllm_config.instance_id} collective_rpc dispatch sent_method={sent_method if isinstance(sent_method, str) else 'Callable'}",
            flush=True,
        )
        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get([worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        print(
            f"instance_id: {self.vllm_config.instance_id} collective_rpc return sent_method={sent_method if isinstance(sent_method, str) else 'Callable'}",
            flush=True,
        )
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(
        self,
        config: DictConfig,
        vllm_dp_size: int,
        vllm_dp_rank: int,
        wg_prefix: str,
        cuda_visible_devices: str | None = None,
        worker_local_sidecar: bool = False,
        worker_names: list[str] | None = None,
        ray_address: str | None = None,
    ):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        self.cuda_visible_devices = cuda_visible_devices
        self.engine: AsyncLLM = None
        self.openai_serving_chat = None
        self.engine_initialized = False
        self.engine_initializing = False
        self.engine_init_error: str | None = None
        self.engine_mode = "v1_async_llm"
        self.enable_sleep_mode = False
        self.worker_local_sidecar = worker_local_sidecar
        self.vllm_use_v1_forced_off = False
        self.worker_names = list(worker_names or [])
        self.ray_address = ray_address
        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.setdefault("VLLM_USE_V1", "1")
        if self.worker_local_sidecar:
            try:
                import torch

                is_rocm = getattr(torch.version, "hip", None) is not None
            except Exception:
                is_rocm = False
            if is_rocm and os.environ.get("VLLM_USE_V1", "1") != "0":
                logger.warning(
                    "Worker-local async vLLM sidecar on ROCm forces VLLM_USE_V1=0 to avoid the "
                    "ROCm Triton/V1 init_worker get_state_cls failure."
                )
                os.environ["VLLM_USE_V1"] = "0"
                self.vllm_use_v1_forced_off = True
        super().__init__(force_dedicated_loop=worker_local_sidecar)

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        if self.cuda_visible_devices is None:
            raise ValueError("AsyncvLLMServer.cuda_visible_devices is not set")
        print(
            f"[AsyncvLLMServer] init_engine start dp_rank={self.vllm_dp_rank} wg_prefix={self.wg_prefix} cuda_visible_devices={self.cuda_visible_devices}",
            flush=True,
        )
        self.engine_initialized = False
        self.engine_initializing = True
        self.engine_init_error = None
        config = self.config
        try:
            model_path = config.model.path
            model_name = "/".join(model_path.split("/")[-2:])
            local_path = copy_to_local(model_path)
            print(
                f"[AsyncvLLMServer] init_engine model_resolved dp_rank={self.vllm_dp_rank} local_path={local_path}",
                flush=True,
            )
            trust_remote_code = config.model.get("trust_remote_code", False)
            config = config.rollout
            self.enable_sleep_mode = bool(getattr(config, "enable_sleep_mode", False))

            tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
            max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
            max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
            max_model_len = int(max_model_len)

            # Override default generation config from hugging face model config,
            # user can still override them by passing kwargs in each request.
            kwargs = dict(
                n=1,
                logprobs=0,
                repetition_penalty=1.0,
                max_new_tokens=config.response_length,
            )
            for k in config.keys():
                if hasattr(SamplingParams(), str(k)):
                    kwargs[k] = config.get(k)
            print(f"override_generation_config: {kwargs}")

            engine_args = AsyncEngineArgs(
                model=local_path,
                # ROCm async rollout currently launches vLLM inside a CPU Ray actor.
                # Sleep mode probes GPU properties during config creation and can fail
                # before the external executor takes ownership of the device.
                enable_sleep_mode=False,
                override_generation_config=kwargs,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                max_model_len=max_model_len,
                load_format="auto",
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_prefix_caching=True,
                trust_remote_code=trust_remote_code,
                seed=self.vllm_dp_rank,
            )

            # init async llm engine
            print(f"[AsyncvLLMServer] init_engine create_engine_config dp_rank={self.vllm_dp_rank}", flush=True)
            if self.worker_names:
                os.environ["VERL_VLLM_WORKER_NAMES_JSON"] = json.dumps(self.worker_names)
            if self.ray_address:
                os.environ["VERL_VLLM_RAY_ADDRESS"] = self.ray_address
            namespace = ray.get_runtime_context().namespace
            if namespace:
                os.environ["VERL_VLLM_RAY_NAMESPACE"] = namespace
            use_v1 = os.environ.get("VLLM_USE_V1", "1") != "0"
            if use_v1:
                vllm_config = engine_args.create_engine_config()
                vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
                print(f"[AsyncvLLMServer] init_engine from_vllm_config dp_rank={self.vllm_dp_rank}", flush=True)
                self.engine = AsyncLLM.from_vllm_config(vllm_config)
                model_config = self.engine.model_config
                self.engine_mode = "v1_async_llm"
            else:
                print(f"[AsyncvLLMServer] init_engine from_engine_args_v0 dp_rank={self.vllm_dp_rank}", flush=True)
                self.engine = AsyncLLMEngine.from_engine_args(
                    engine_args,
                    usage_context=UsageContext.OPENAI_API_SERVER,
                )
                model_config = self.engine.get_model_config()
                if inspect.isawaitable(model_config):
                    model_config = await model_config
                self.engine_mode = "v0_async_llm_engine"
            print(f"[AsyncvLLMServer] init_engine engine_created dp_rank={self.vllm_dp_rank} mode={self.engine_mode}", flush=True)

            # build serving chat
            BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
            models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models,
                "assistant",
                request_logger=RequestLogger(max_log_len=4096),
                chat_template=None,
                chat_template_content_format="auto",
                enable_auto_tools=True,
                tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
            )
            print(f"[AsyncvLLMServer] init_engine serving_chat_ready dp_rank={self.vllm_dp_rank}", flush=True)
            self.engine_initialized = True
            logger.info(
                "Initialized AsyncvLLMServer dp_rank=%s wg_prefix=%s cuda_visible_devices=%s address=%s:%s",
                self.vllm_dp_rank,
                self.wg_prefix,
                self.cuda_visible_devices,
                self.address,
                self.port,
            )
        except Exception as exc:
            self.engine = None
            self.openai_serving_chat = None
            self.engine_init_error = repr(exc)
            logger.exception(
                "Failed to initialize AsyncvLLMServer dp_rank=%s wg_prefix=%s cuda_visible_devices=%s",
                self.vllm_dp_rank,
                self.wg_prefix,
                self.cuda_visible_devices,
            )
            raise
        finally:
            self.engine_initializing = False

    async def describe_runtime(self) -> Dict[str, Any]:
        return {
            "dp_rank": self.vllm_dp_rank,
            "dp_size": self.vllm_dp_size,
            "wg_prefix": self.wg_prefix,
            "cuda_visible_devices": self.cuda_visible_devices,
            "address": self.address,
            "port": self.port,
            "server_ready": self.server_ready.is_set(),
            "server_ready_metadata": dict(self.server_ready_metadata),
            "engine_initialized": self.engine_initialized,
            "engine_initializing": self.engine_initializing,
            "engine_init_error": self.engine_init_error,
            "worker_local_sidecar": self.worker_local_sidecar,
            "worker_names": list(self.worker_names),
            "ray_address": self.ray_address,
            "loop_mode": getattr(self, "_loop_mode", "unknown"),
            "vllm_use_v1": os.environ.get("VLLM_USE_V1"),
            "vllm_use_v1_forced_off": self.vllm_use_v1_forced_off,
        }

    def _require_engine_initialized(self, action: str) -> None:
        if not self.engine_initialized or self.engine is None:
            raise RuntimeError(
                f"AsyncvLLMServer dp_rank={self.vllm_dp_rank} cannot {action} before init_engine succeeds. "
                f"last_error={self.engine_init_error}"
            )

    async def _invoke_engine_method(self, method_name: str, *args, **kwargs):
        """Handle vLLM engine methods that may be sync, async, or sync wrappers returning awaitables."""
        self._require_engine_initialized(method_name)
        method = getattr(self.engine, method_name)
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def wake_up(self):
        if self.enable_sleep_mode:
            await self._invoke_engine_method("wake_up", tags=["kv_cache", "weights"])

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self._invoke_engine_method("reset_prefix_cache")
        if self.enable_sleep_mode:
            await self._invoke_engine_method("sleep")
