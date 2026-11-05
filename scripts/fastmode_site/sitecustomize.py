"""Fast mode start-up hook (owner 2026-10-17): loaded only when this folder is on
PYTHONPATH (scripts/fast_mode.py sets it) AND ``SPLITSHIELD_FAST_BATCH`` is set.

It overrides the training batch size in ``config.base`` right after that module is
imported, in this process and in every child process, without editing any research
file. The batch size enters the config hash (effective batch size), so fast-mode runs
get their own run ids and never mix with registered runs.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from typing import Any

_BATCH = os.environ.get("SPLITSHIELD_FAST_BATCH")


class _PatchConfig(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name != "config.base":
            return None
        sys.meta_path.remove(self)  # find the real module without recursing
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader: Any = spec.loader
        original_exec = loader.exec_module

        def exec_module(module: Any) -> None:
            original_exec(module)
            from dataclasses import replace

            apply_env = module._apply_env
            batch = int(str(_BATCH))

            def fast_apply_env(runtime: Any) -> Any:
                return replace(apply_env(runtime), batch_size=batch)

            module._apply_env = fast_apply_env

        loader.exec_module = exec_module
        return spec


if _BATCH:
    sys.meta_path.insert(0, _PatchConfig())
