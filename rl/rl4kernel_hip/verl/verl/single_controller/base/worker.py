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
"""
the class for Worker
"""

import logging
import os
import socket
import time
import asyncio
from dataclasses import dataclass
from typing import Dict

import ray

from verl.utils.device import get_torch_device

from .decorator import Dispatch, Execute, register

logger = logging.getLogger(__file__)


def _unwrap_actor_class(cls):
    metadata = getattr(cls, "__ray_metadata__", None)
    if metadata is not None and hasattr(metadata, "modified_class"):
        return metadata.modified_class
    if hasattr(cls, "__ray_actor_class__"):
        return cls.__ray_actor_class__
    return cls


@dataclass
class DistRankInfo:
    tp_rank: int
    dp_rank: int
    pp_rank: int
    cp_rank: int


@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int
    cp_size: int


class WorkerHelper:
    def _get_node_ip(self):
        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                import ray

                return ray._private.services.get_node_ip_address()
            else:
                raise NotImplementedError("WG_BACKEND now just support ray mode.")

        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = get_node_ip_by_sdk()

        host_ip = host_ip_by_env or host_ip_by_sdk
        return host_ip

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip(), str(self._get_free_port())

    def _get_pid(self):
        return os.getpid()


# we assume that in each WorkerGroup, there is a Master Worker
class Worker(WorkerHelper):
    """A distributed worker that handles initialization and configuration for distributed training.

    This class manages worker initialization, configuration, and provides methods for executing
    distributed operations. It handles communication settings, device configuration, and worker
    metadata management.
    """

    fused_worker_attr_name = "fused_worker_dict"

    def __new__(cls, *args, **kwargs):
        """Create a new Worker instance with proper initialization based on environment settings."""
        instance = super().__new__(cls)

        # note that here we use int to distinguish
        disable_worker_init = int(os.environ.get("DISABLE_WORKER_INIT", 0))
        if disable_worker_init:
            return instance

        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
        if None not in [rank, worker_group_prefix] and "ActorClass(" not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        """Configure worker settings before initialization.

        Args:
            register_center_name (str):
                Name of the register center Ray actor for worker coordination
            rank (int):
                Rank of the worker in the distributed setup
        """
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center

                self.register_center = create_worker_group_register_center(name=register_center_name, info=rank_zero_info)

            os.environ.update(rank_zero_info)
        else:
            self.register_center = ray.get_actor(register_center_name)

        # set worker info for node affinity scheduling
        ray.get(self.register_center.set_worker_info.remote(rank, ray.get_runtime_context().get_node_id()))

    @classmethod
    def env_keys(cls):
        """The keys of the environment variables that are used to configure the Worker."""
        return ["WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"]

    def __init__(self, cuda_visible_devices=None) -> None:
        """Initialize the worker with environment settings and device configuration.

        Args:
            cuda_visible_devices (str, optional):
                CUDA visible devices configuration. Defaults to None.
        """
        # construct a meta from environment variable. Note that the import must be inside the class because it is executed remotely
        import os

        self._setup_env_cuda_visible_devices()

        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            "_world_size": world_size,
            "_rank": rank,
            "_local_world_size": local_world_size,
            "_local_rank": local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        if cuda_visible_devices is not None:
            store["_cuda_visible_devices"] = cuda_visible_devices

        self._configure_with_store(store=store)

        self.fused_worker_dict = {}
        self._spawned_sidecars = {}

    def get_fused_worker_by_name(self, worker_name: str):
        """Get a fused worker by its name.

        Args:
            worker_name (str):
                Name of the worker to retrieve
        """
        return self.fused_worker_dict.get(worker_name, None)

    def _setup_env_cuda_visible_devices(self):
        from verl.utils.ray_utils import ray_noset_visible_devices

        is_ray_noset_visible_devices = ray_noset_visible_devices()

        # Prevent use of clashing `{CUDA/HIP/ROCR}_VISIBLE_DEVICES``
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if hip_val:
            # Switch the use of HIP_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES for consistency.
            # Make sure that the HIP_VISIBLE_DEVICES is set to the same value as CUDA_VISIBLE_DEVICES
            # at this point.
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            hip_val = None
            if cuda_val:
                assert val == cuda_val, f"Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values found: {val} and {cuda_val}."
            else:
                cuda_val = val
                os.environ["CUDA_VISIBLE_DEVICES"] = val

        if rocr_val:
            # You must take care if both HIP/CUDA and ROCR env vars are set as they have
            # different meanings. Both env vars accept either a list of ints or a
            # list of UUIDs. The ROCR env var is processed first which then reduces
            # the number of GPUs that HIP can select from.
            # https://github.com/pytorch/pytorch/pull/144026
            # To avoid the complexity of this, we simply gives out error if both are set
            # (Also to keep consistency with ray's practice with 2.45.0).
            # Otherwise, we will set ROCR_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES
            # and remove ROCR_VISIBLE_DEVICES.
            if cuda_val:
                raise ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")

            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val
            rocr_val = None

        if is_ray_noset_visible_devices:
            # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
            # environment variable for each actor, unless
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set,
            # so we need to set local rank when the flag is set.
            local_rank = os.environ.get("RAY_LOCAL_RANK")
            os.environ["LOCAL_RANK"] = local_rank
            get_torch_device().set_device(int(local_rank))

    def _configure_with_store(self, store: Dict):
        """
        This function should only be called inside by WorkerGroup
        """
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        self.__dict__.update(store_env_dict)  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        for key in type(self).env_keys():
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""

    def get_master_addr_port(self):
        """Get the master address and port for distributed communication."""
        return self._master_addr, self._master_port

    def get_node_id(self):
        """Get the Ray node id for colocated sidecar actors."""
        import ray

        return ray.get_runtime_context().get_node_id()

    def get_cuda_visible_devices(self):
        """Get the CUDA visible devices configuration."""
        import os

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        return cuda_visible_devices

    def _require_sidecar_cuda_visible_devices(self) -> str:
        cuda_visible_devices = str(self.get_cuda_visible_devices())
        if cuda_visible_devices in {"", "not set"}:
            raise ValueError("CUDA_VISIBLE_DEVICES is not set on the rollout worker.")
        return cuda_visible_devices

    def _build_sidecar_runtime_env(self, server_runtime_env: Dict | None, cuda_visible_devices: str) -> Dict:
        runtime_env = dict(server_runtime_env or {})
        env_vars = dict(runtime_env.get("env_vars", {}))
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        runtime_env["env_vars"] = env_vars
        return runtime_env

    def _replace_existing_sidecar(self, server_name: str) -> None:
        existing_local = self._spawned_sidecars.pop(server_name, None)
        if existing_local is not None:
            logger.warning("Replacing stale local async sidecar name=%s", server_name)
            shutdown = getattr(existing_local, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    logger.warning("Failed to shutdown local async sidecar name=%s: %s", server_name, exc)
        try:
            existing = ray.get_actor(server_name)
        except ValueError:
            existing = None
        if existing is not None:
            logger.warning("Replacing stale async sidecar actor name=%s", server_name)
            ray.kill(existing)

    def _get_spawned_sidecar(self, server_name: str):
        server = self._spawned_sidecars.get(server_name)
        if server is None:
            raise KeyError(f"Unknown async sidecar: {server_name}")
        return server

    def _run_spawned_sidecar(self, server_name: str, method_name: str, *args, timeout_s: float | None = None, **kwargs):
        server = self._get_spawned_sidecar(server_name)
        method = getattr(server, method_name)
        print(
            f"[worker-owned-sidecar] start server_name={server_name} method={method_name} timeout_s={timeout_s}",
            flush=True,
        )
        logger.info(
            "Calling local async sidecar server_name=%s method=%s timeout_s=%s",
            server_name,
            method_name,
            timeout_s,
        )
        start = time.time()
        try:
            result = server.run_coroutine_sync(method(*args, **kwargs), timeout_s=timeout_s)
        except Exception:
            print(
                f"[worker-owned-sidecar] fail server_name={server_name} method={method_name} elapsed_s={time.time() - start:.2f}",
                flush=True,
            )
            logger.exception(
                "Local async sidecar call failed server_name=%s method=%s elapsed_s=%.2f",
                server_name,
                method_name,
                time.time() - start,
            )
            raise
        print(
            f"[worker-owned-sidecar] done server_name={server_name} method={method_name} elapsed_s={time.time() - start:.2f}",
            flush=True,
        )
        logger.info(
            "Local async sidecar call finished server_name=%s method=%s elapsed_s=%.2f",
            server_name,
            method_name,
            time.time() - start,
        )
        return result

    def spawn_colocated_async_server(
        self,
        rollout_backend: str,
        config,
        dp_size: int,
        dp_rank: int,
        wg_prefix: str,
        server_name: str,
        worker_names: list[str] | None = None,
        ray_address: str | None = None,
        server_runtime_env: Dict | None = None,
    ):
        """Spawn an async rollout sidecar actor with this worker's GPU context.

        This is the native path for sidecar creation: the rollout worker that already
        owns the Ray GPU context brokers the sidecar actor spawn, instead of asking a
        CPU-only coordinator to reconstruct GPU visibility heuristically.
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from verl.workers.rollout.async_server import async_server_class

        cuda_visible_devices = self._require_sidecar_cuda_visible_devices()
        runtime_env = self._build_sidecar_runtime_env(server_runtime_env, cuda_visible_devices)
        node_id = self.get_node_id()
        self._replace_existing_sidecar(server_name)

        server_class = async_server_class(rollout_backend)
        logger.info(
            "Spawning async sidecar rank=%s node_id=%s server_name=%s rollout_backend=%s cuda_visible_devices=%s",
            getattr(self, "_rank", "unknown"),
            node_id,
            server_name,
            rollout_backend,
            cuda_visible_devices,
        )
        if rollout_backend == "vllm":
            env_vars = runtime_env.get("env_vars", {})
            for key, value in env_vars.items():
                os.environ[str(key)] = str(value)
            local_server_class = _unwrap_actor_class(server_class)
            server = local_server_class(
                config,
                dp_size,
                dp_rank,
                wg_prefix,
                cuda_visible_devices=cuda_visible_devices,
                worker_local_sidecar=True,
                worker_names=list(worker_names or []),
                ray_address=ray_address,
            )
            self._spawned_sidecars[server_name] = server
            logger.info("Spawned local worker-owned async sidecar server_name=%s sidecar=%s", server_name, server)
            return {"server_name": server_name, "mode": "worker_owned_local_sidecar"}

        options = {
            "scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            "runtime_env": runtime_env,
            "name": server_name,
        }
        remote_args = [config, dp_size, dp_rank, wg_prefix]
        server = server_class.options(**options).remote(*remote_args)
        self._spawned_sidecars[server_name] = server
        logger.info("Spawned async sidecar server_name=%s actor=%s", server_name, server)
        return server

    def get_colocated_async_server_address(self, server_name: str) -> str:
        timeout_s = float(os.environ.get("VERL_LOCAL_ASYNC_SIDECAR_ADDRESS_TIMEOUT_S", "60"))
        return self._run_spawned_sidecar(server_name, "get_server_address", timeout_s=timeout_s)

    async def init_colocated_async_server(self, server_name: str):
        timeout_s = float(os.environ.get("VERL_LOCAL_ASYNC_SIDECAR_INIT_TIMEOUT_S", "1800"))
        return await asyncio.to_thread(self._run_spawned_sidecar, server_name, "init_engine", timeout_s=timeout_s)

    def describe_colocated_async_server(self, server_name: str) -> Dict:
        timeout_s = float(os.environ.get("VERL_LOCAL_ASYNC_SIDECAR_DESCRIBE_TIMEOUT_S", "30"))
        return self._run_spawned_sidecar(server_name, "describe_runtime", timeout_s=timeout_s)

    def wake_up_colocated_async_server(self, server_name: str):
        timeout_s = float(os.environ.get("VERL_LOCAL_ASYNC_SIDECAR_WAKE_TIMEOUT_S", "300"))
        return self._run_spawned_sidecar(server_name, "wake_up", timeout_s=timeout_s)

    def sleep_colocated_async_server(self, server_name: str):
        timeout_s = float(os.environ.get("VERL_LOCAL_ASYNC_SIDECAR_SLEEP_TIMEOUT_S", "300"))
        return self._run_spawned_sidecar(server_name, "sleep", timeout_s=timeout_s)

    def stop_colocated_async_server(self, server_name: str) -> bool:
        server = self._spawned_sidecars.pop(server_name, None)
        if server is None:
            return False
        shutdown = getattr(server, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return True

    @property
    def world_size(self):
        """Get the total number of workers in the distributed setup."""
        return self._world_size

    @property
    def rank(self):
        """Get the rank of this worker in the distributed setup."""
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        """Execute a function in rank zero execution mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        result = func(*args, **kwargs)
        return result
