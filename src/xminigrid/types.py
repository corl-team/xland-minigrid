from typing import TypeAlias

import jax
import jax.numpy as jnp
from flax import struct

from .core.constants import TILES_REGISTRY, Colors, Tiles


class RuleSet(struct.PyTreeNode):
    goal: jax.Array
    rules: jax.Array
    init_tiles: jax.Array


GridState: TypeAlias = jax.Array


class AgentState(struct.PyTreeNode):
    position: jax.Array = jnp.asarray((0, 0))
    direction: jax.Array = jnp.asarray(0)
    pocket: jax.Array = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]


class EnvCarry(struct.PyTreeNode):
    ...


class State(struct.PyTreeNode):
    key: jax.random.PRNGKey
    step_num: jax.Array

    grid: jax.Array
    agent: AgentState
    goal_encoding: jax.Array
    rule_encoding: jax.Array

    carry: EnvCarry


class StepType(jnp.uint8):
    FIRST: int = jnp.asarray(0, dtype=jnp.uint8)
    MID: int = jnp.asarray(1, dtype=jnp.uint8)
    LAST: int = jnp.asarray(2, dtype=jnp.uint8)


class TimeStep(struct.PyTreeNode):
    state: State

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST
