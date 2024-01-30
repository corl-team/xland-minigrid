from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Type

from .environment import Environment, EnvParams


@dataclass
class EnvSpec:
    id: str
    entry_point: str
    kwargs: dict = field(default_factory=dict)


_REGISTRY: dict[str, EnvSpec] = {}


# TODO: add ability to apply wrappers
def register(
    id: str,
    entry_point: str,
    **kwargs,
) -> None:
    if id in _REGISTRY:
        raise ValueError("Environment with such id is already registered. Please choose another one.")

    if not id.startswith("XLand-MiniGrid"):
        if not id.startswith("MiniGrid"):
            raise ValueError("Invalid id format. Should start from XLand-MiniGrid or MiniGrid.")

    env_spec = EnvSpec(id=id, entry_point=entry_point, kwargs=kwargs)
    _REGISTRY[id] = env_spec


def load(name: str) -> Type[Environment]:
    mod_name, env_name = name.split(":")
    mod = importlib.import_module(mod_name)
    env_constructor = getattr(mod, env_name)
    return env_constructor


def make(id: str, **kwargs: Any) -> tuple[Environment, EnvParams]:
    if id not in _REGISTRY:
        raise ValueError(f"Unregistered environment. Available environments: {', '.join(registered_environments())}")

    env_spec = _REGISTRY[id]
    env_constructor = load(env_spec.entry_point)

    env = env_constructor()
    env_params = env.default_params(**env_spec.kwargs)
    env_params = env_params.replace(**kwargs)

    return env, env_params


def registered_environments() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())
