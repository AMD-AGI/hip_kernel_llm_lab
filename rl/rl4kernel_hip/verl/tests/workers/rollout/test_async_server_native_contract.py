# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.single_controller.base.worker import Worker
from verl.workers.rollout.async_server import AsyncLLMServerManager


class RemoteCall:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeServerClass:
    def __init__(self):
        self.options_calls = []
        self.remote_calls = []

    def options(self, **options):
        self.options_calls.append(options)
        owner = self

        class _BoundOptions:
            def remote(self, *args, **kwargs):
                handle = SimpleNamespace(actor_name=options.get("name", "unnamed"))
                owner.remote_calls.append(
                    {
                        "args": args,
                        "kwargs": kwargs,
                        "handle": handle,
                    }
                )
                return handle

        return _BoundOptions()


class FakeServerHandle:
    def __init__(self, address: str, *, init_exception: Exception | None = None, runtime_info: dict | None = None):
        self.address = address
        self.init_exception = init_exception
        self.runtime_info = runtime_info or {
            "engine_initialized": True,
            "cuda_visible_devices": "0",
            "server_ready": True,
        }
        self.wake_up = RemoteCall(lambda: True)
        self.sleep = RemoteCall(lambda: True)
        self.get_server_address = RemoteCall(lambda: self.address)
        self.init_engine = RemoteCall(self._init_engine)
        self.describe_runtime = RemoteCall(lambda: self.runtime_info)

    def _init_engine(self):
        if self.init_exception is not None:
            raise self.init_exception
        return True


class FakeWorkerActor:
    def __init__(self, servers: list[FakeServerHandle]):
        self._servers = list(servers)
        self.spawn_calls = []
        self.stop_calls = []
        self._spawned = {}
        self.spawn_colocated_async_server = RemoteCall(self._spawn)
        self.get_colocated_async_server_address = RemoteCall(self._get_address)
        self.init_colocated_async_server = RemoteCall(self._init_server)
        self.describe_colocated_async_server = RemoteCall(self._describe_server)
        self.wake_up_colocated_async_server = RemoteCall(self._wake_up)
        self.sleep_colocated_async_server = RemoteCall(self._sleep)
        self.stop_colocated_async_server = RemoteCall(self._stop)

    def _spawn(self, **kwargs):
        self.spawn_calls.append(kwargs)
        assert self._servers, "No fake servers left to spawn."
        server = self._servers.pop(0)
        self._spawned[kwargs["server_name"]] = server
        return {"server_name": kwargs["server_name"], "mode": "worker_owned_local_sidecar"}

    def _get_server(self, server_name: str) -> FakeServerHandle:
        return self._spawned[server_name]

    def _get_address(self, server_name: str):
        return self._get_server(server_name).address

    def _init_server(self, server_name: str):
        return self._get_server(server_name)._init_engine()

    def _describe_server(self, server_name: str):
        return self._get_server(server_name).runtime_info

    def _wake_up(self, server_name: str):
        return True

    def _sleep(self, server_name: str):
        return True

    def _stop(self, server_name: str):
        self.stop_calls.append(server_name)
        self._spawned.pop(server_name, None)
        return True


class FakeRegisterCenter:
    def __init__(self, worker_info):
        self.get_worker_info = RemoteCall(lambda: worker_info)


def _make_bare_worker(cuda_visible_devices: str = "3,5", node_id: str = "node-1", rank: int = 7) -> Worker:
    worker = object.__new__(Worker)
    worker._rank = rank
    worker._spawned_sidecars = {}
    worker.get_cuda_visible_devices = lambda: cuda_visible_devices
    worker.get_node_id = lambda: node_id
    return worker


def _ray_get_identity(value, timeout=None):
    if isinstance(value, list):
        return [_ray_get_identity(item, timeout=timeout) for item in value]
    return value


def _minimal_manager_config():
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "tensor_model_parallel_size": 1,
                    "name": "vllm",
                }
            }
        }
    )


def _fake_init_chat_scheduler(self):
    self.chat_scheduler = SimpleNamespace()
    self.chat_scheduler_loop = object()
    self.chat_scheduler_ready.set()


class FakeLocalVLLMServer:
    def __init__(
        self,
        config,
        dp_size,
        dp_rank,
        wg_prefix,
        cuda_visible_devices=None,
        worker_local_sidecar=False,
        worker_names=None,
        ray_address=None,
    ):
        self.config = config
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.wg_prefix = wg_prefix
        self.cuda_visible_devices = cuda_visible_devices
        self.worker_local_sidecar = worker_local_sidecar
        self.worker_names = list(worker_names or [])
        self.ray_address = ray_address

    def shutdown(self):
        return None


def test_worker_spawn_uses_worker_cuda_visible_devices_and_replaces_stale_actor(monkeypatch):
    worker = _make_bare_worker(cuda_visible_devices="2,4")
    stale_actor = SimpleNamespace(name="stale-sidecar")
    kill_calls = []

    monkeypatch.setattr("verl.single_controller.base.worker.ray.get_actor", lambda name: stale_actor)
    monkeypatch.setattr("verl.single_controller.base.worker.ray.kill", lambda actor: kill_calls.append(actor))
    monkeypatch.setattr("verl.workers.rollout.async_server.async_server_class", lambda backend: SimpleNamespace(__ray_metadata__=SimpleNamespace(modified_class=FakeLocalVLLMServer)))

    server = worker.spawn_colocated_async_server(
        rollout_backend="vllm",
        config={"cfg": "value"},
        dp_size=2,
        dp_rank=1,
        wg_prefix="wgtest",
        server_name="wgtest_async_llm_server_1",
        server_runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
    )

    assert kill_calls == [stale_actor]
    assert server == {"server_name": "wgtest_async_llm_server_1", "mode": "worker_owned_local_sidecar"}
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].cuda_visible_devices == "2,4"
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].dp_rank == 1
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].wg_prefix == "wgtest"
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].worker_local_sidecar is True
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].worker_names == []
    assert worker._spawned_sidecars["wgtest_async_llm_server_1"].ray_address is None


def test_worker_spawn_requires_worker_side_visible_devices(monkeypatch):
    worker = _make_bare_worker(cuda_visible_devices="not set")
    monkeypatch.setattr("verl.single_controller.base.worker.ray.get_actor", lambda name: (_ for _ in ()).throw(ValueError(name)))

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is not set on the rollout worker"):
        worker.spawn_colocated_async_server(
            rollout_backend="vllm",
            config={"cfg": "value"},
            dp_size=1,
            dp_rank=0,
            wg_prefix="wgtest",
            server_name="wgtest_async_llm_server_0",
        )


def test_async_server_manager_retries_until_engine_ready(monkeypatch):
    config = _minimal_manager_config()
    first_server = FakeServerHandle(
        "127.0.0.1:8001",
        runtime_info={
            "engine_initialized": False,
            "cuda_visible_devices": "0",
            "server_ready": True,
        },
    )
    second_server = FakeServerHandle(
        "127.0.0.1:8002",
        runtime_info={
            "engine_initialized": True,
            "cuda_visible_devices": "0",
            "server_ready": True,
        },
    )
    fake_worker = FakeWorkerActor([first_server, second_server])
    fake_group = SimpleNamespace(world_size=1, name_prefix="wgtest", workers=[fake_worker], worker_names=["wgtestWorkerDict_0:0"])
    fake_register_center = FakeRegisterCenter(["node-a"])
    kill_calls = []

    monkeypatch.setattr(AsyncLLMServerManager, "_init_chat_scheduler", _fake_init_chat_scheduler)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_actor", lambda name: fake_register_center)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get", _ray_get_identity)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_runtime_context", lambda: SimpleNamespace(gcs_address="ray://fake-address"))
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.kill", lambda handle: kill_calls.append(handle))

    manager = AsyncLLMServerManager(config=config, worker_group=fake_group)

    assert fake_worker.spawn_calls[0]["server_name"] == "wgtest_async_llm_server_0"
    assert fake_worker.spawn_calls[0]["worker_names"] == ["wgtestWorkerDict_0:0"]
    assert fake_worker.spawn_calls[0]["ray_address"] == "ray://fake-address"
    assert manager.sidecar_slots[0].spawn_attempts == 2
    assert manager.server_addresses == ["127.0.0.1:8002"]
    assert manager.sidecar_slots[0].engine_initialized is True
    assert fake_worker.stop_calls == ["wgtest_async_llm_server_0"]
    assert kill_calls == []
    runtime = _ray_get_identity(manager.async_llm_servers[0].describe_runtime.remote())
    assert runtime["engine_initialized"] is True


def test_async_server_manager_stops_after_bounded_retries(monkeypatch):
    config = _minimal_manager_config()
    bad_servers = [
        FakeServerHandle("127.0.0.1:8101", runtime_info={"engine_initialized": False}),
        FakeServerHandle("127.0.0.1:8102", runtime_info={"engine_initialized": False}),
    ]
    fake_worker = FakeWorkerActor(bad_servers)
    fake_group = SimpleNamespace(world_size=1, name_prefix="wgtest", workers=[fake_worker], worker_names=["wgtestWorkerDict_0:0"])
    fake_register_center = FakeRegisterCenter(["node-a"])
    kill_calls = []

    monkeypatch.setenv("VERL_ASYNC_SERVER_MAX_RETRIES", "2")
    monkeypatch.setattr(AsyncLLMServerManager, "_init_chat_scheduler", _fake_init_chat_scheduler)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_actor", lambda name: fake_register_center)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get", _ray_get_identity)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_runtime_context", lambda: SimpleNamespace(gcs_address="ray://fake-address"))
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.kill", lambda handle: kill_calls.append(handle))

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        AsyncLLMServerManager(config=config, worker_group=fake_group)

    assert fake_worker.stop_calls == ["wgtest_async_llm_server_0", "wgtest_async_llm_server_0"]
    assert len(kill_calls) == 0


def test_async_server_manager_public_methods_fail_if_slot_not_ready(monkeypatch):
    config = _minimal_manager_config()
    ready_server = FakeServerHandle("127.0.0.1:8200")
    fake_worker = FakeWorkerActor([ready_server])
    fake_group = SimpleNamespace(world_size=1, name_prefix="wgtest", workers=[fake_worker], worker_names=["wgtestWorkerDict_0:0"])
    fake_register_center = FakeRegisterCenter(["node-a"])

    monkeypatch.setattr(AsyncLLMServerManager, "_init_chat_scheduler", _fake_init_chat_scheduler)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_actor", lambda name: fake_register_center)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get", _ray_get_identity)
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.get_runtime_context", lambda: SimpleNamespace(gcs_address="ray://fake-address"))
    monkeypatch.setattr("verl.workers.rollout.async_server.ray.kill", lambda handle: None)

    manager = AsyncLLMServerManager(config=config, worker_group=fake_group)
    manager.sidecar_slots[0].engine_initialized = False

    with pytest.raises(RuntimeError, match="not ready"):
        manager.wake_up()
