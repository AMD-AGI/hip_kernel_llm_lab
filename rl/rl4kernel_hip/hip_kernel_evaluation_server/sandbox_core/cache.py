from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import torch


CACHE_SCHEMA_VERSION = 2


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def strip_candidate_hash_suffix(kernel_name: str) -> str:
    if not kernel_name:
        return ""
    parts = kernel_name.split("_")
    if len(parts) > 1:
        last_part = parts[-1]
        if len(last_part) == 8 and all(c in "0123456789abcdef" for c in last_part.lower()):
            return "_".join(parts[:-1])
    return kernel_name


def _stable_payload_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReferenceGoldenKey:
    logical_kernel_name: str
    driver_kind: str
    hip_ref_sha256: str
    pytorch_functional_sha256: str
    pytorch_module_sha256: str
    template_bundle_sha256: str
    arch: str
    software_stack_fingerprint: Dict[str, Any]
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_payload_hash(asdict(self))


@dataclass(frozen=True)
class ReferenceCompileArtifactKey:
    logical_kernel_name: str
    driver_kind: str
    hip_ref_sha256: str
    pytorch_functional_sha256: str
    pytorch_module_sha256: str
    template_bundle_sha256: str
    arch: str
    compiler_identity: Dict[str, Any]
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_payload_hash(asdict(self))


@dataclass(frozen=True)
class ReferencePerfKey:
    logical_kernel_name: str
    driver_kind: str
    hip_ref_sha256: str
    pytorch_functional_sha256: str
    pytorch_module_sha256: str
    template_bundle_sha256: str
    arch: str
    perf_iterations: int
    runtime_fingerprint: Dict[str, Any]
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def cache_id(self) -> str:
        return _stable_payload_hash(asdict(self))


class ReferenceCache:
    def __init__(self, cache_root: str):
        self.cache_root = os.path.abspath(cache_root)

    def _compile_dir(self, key: ReferenceCompileArtifactKey) -> str:
        return os.path.join(self.cache_root, "compile", key.cache_id)

    def _golden_dir(self, key: ReferenceGoldenKey) -> str:
        return os.path.join(self.cache_root, "golden", key.cache_id)

    def _perf_dir(self, key: ReferencePerfKey) -> str:
        return os.path.join(self.cache_root, "perf", key.cache_id)

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".tmp", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_write_text(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".txt", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_write_torch(path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=os.path.dirname(path))
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _compiled_library_exists(build_directory: str, module_name: str) -> bool:
        if not build_directory or not os.path.isdir(build_directory):
            return False
        for root, _, files in os.walk(build_directory):
            for filename in files:
                lower = filename.lower()
                if not lower.endswith((".so", ".pyd", ".dll", ".dylib")):
                    continue
                if module_name in filename:
                    return True
        return False

    def compile_artifact_layout(self, key: ReferenceCompileArtifactKey, *, module_name: str) -> Dict[str, Any]:
        artifact_root = self._compile_dir(key)
        source_dir = os.path.join(artifact_root, "src")
        build_dir = os.path.join(artifact_root, "build")
        source_path = os.path.join(source_dir, "reference_kernel.hip")
        meta_path = os.path.join(artifact_root, "meta.json")
        return {
            "artifact_root": artifact_root,
            "source_dir": source_dir,
            "source_path": source_path,
            "build_directory": build_dir,
            "meta_path": meta_path,
            "module_name": module_name,
        }

    def ensure_compile_source(
        self,
        key: ReferenceCompileArtifactKey,
        *,
        module_name: str,
        source_text: str,
    ) -> Dict[str, Any]:
        layout = self.compile_artifact_layout(key, module_name=module_name)
        source_path = layout["source_path"]
        should_write = True
        if os.path.exists(source_path):
            try:
                with open(source_path, "r", encoding="utf-8") as handle:
                    should_write = handle.read() != source_text
            except Exception:
                should_write = True
        if should_write:
            self._atomic_write_text(source_path, source_text)
        os.makedirs(layout["build_directory"], exist_ok=True)
        return layout

    def load_compile_artifact(self, key: ReferenceCompileArtifactKey) -> Optional[Dict[str, Any]]:
        artifact_root = self._compile_dir(key)
        meta_path = os.path.join(artifact_root, "meta.json")
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        source_path = payload.get("source_path")
        build_directory = payload.get("build_directory")
        module_name = payload.get("module_name")
        if not source_path or not os.path.exists(source_path):
            return None
        if not build_directory or not module_name:
            return None
        if not self._compiled_library_exists(build_directory, module_name):
            return None
        return payload

    def store_compile_artifact(self, key: ReferenceCompileArtifactKey, meta: Dict[str, Any]) -> None:
        artifact_dir = self._compile_dir(key)
        os.makedirs(artifact_dir, exist_ok=True)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "created_at_epoch": time.time(),
            **meta,
        }
        self._atomic_write_json(os.path.join(artifact_dir, "meta.json"), payload)

    def load_golden(self, key: ReferenceGoldenKey) -> Optional[Tuple[Any, Dict[str, Any]]]:
        golden_dir = self._golden_dir(key)
        golden_path = os.path.join(golden_dir, "golden.pt")
        meta_path = os.path.join(golden_dir, "meta.json")
        if not (os.path.exists(golden_path) and os.path.exists(meta_path)):
            return None
        try:
            golden = torch.load(golden_path, map_location="cpu")
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            return golden, meta
        except Exception:
            return None

    def store_golden(self, key: ReferenceGoldenKey, golden: Any, meta: Dict[str, Any]) -> None:
        golden_dir = self._golden_dir(key)
        os.makedirs(golden_dir, exist_ok=True)
        payload = {"cache_schema_version": CACHE_SCHEMA_VERSION, **meta}
        self._atomic_write_torch(os.path.join(golden_dir, "golden.pt"), golden)
        self._atomic_write_json(os.path.join(golden_dir, "meta.json"), payload)

    def load_perf(self, key: ReferencePerfKey, ttl_s: int = 0) -> Optional[Dict[str, Any]]:
        perf_dir = self._perf_dir(key)
        perf_path = os.path.join(perf_dir, "perf.json")
        if not os.path.exists(perf_path):
            return None
        try:
            with open(perf_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        if ttl_s and payload.get("created_at_epoch"):
            if (time.time() - float(payload["created_at_epoch"])) > ttl_s:
                return None
        return payload

    def store_perf(self, key: ReferencePerfKey, perf_ms: float, meta: Dict[str, Any]) -> None:
        perf_dir = self._perf_dir(key)
        os.makedirs(perf_dir, exist_ok=True)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "reference_perf_ms": float(perf_ms),
            "created_at_epoch": time.time(),
            **meta,
        }
        self._atomic_write_json(os.path.join(perf_dir, "perf.json"), payload)
