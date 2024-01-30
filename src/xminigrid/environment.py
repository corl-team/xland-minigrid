from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.random import KeyArray

from .core.actions import take_action
from .core.constants import NUM_ACTIONS, NUM_LAYERS
from .core.goals import check_goal
from .core.observation import transparent_field_of_view
from .core.rules import check_rule
from .rendering.rgb_render import render as rgb_render
from .rendering.text_render import render as text_render
from .types import State, StepType, TimeStep


class EnvParams(struct.PyTreeNode):
    # WARN: pytree_node=False, so you CAN NOT vmap on them!
    # You can add pytree node params, but be careful and
    # test that your code will work under jit.
    # Spoiler: probably it will not :(
    height: int = struct.field(pytree_node=False, default=9)
    width: int = struct.field(pytree_node=False, default=9)
    view_size: int = struct.field(pytree_node=False, default=7)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")


# TODO: add generic type hints (on env params)
class Environment:
    def default_params(self, **kwargs: Any) -> EnvParams:
        return EnvParams().replace(**kwargs)

    def num_actions(self, params: EnvParams) -> int:
        return int(NUM_ACTIONS)

    def observation_shape(self, params: EnvParams) -> tuple[int, int, int]:
        return params.view_size, params.view_size, NUM_LAYERS

    # TODO: NOT sure that this should be hardcoded like that...
    def time_limit(self, params: EnvParams) -> int:
        return 3 * params.height * params.width

    def _generate_problem(self, params: EnvParams, key: KeyArray) -> State:
        return NotImplemented

    def reset(self, params: EnvParams, key: KeyArray) -> TimeStep:
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view(state.grid, state.agent, params.view_size, params.view_size),
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParams, timestep: TimeStep, action: int) -> TimeStep:
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        new_observation = transparent_field_of_view(new_state.grid, new_state.agent, params.view_size, params.view_size)

        # checking for termination or truncation, choosing step type
        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)
        truncated = jnp.equal(new_state.step_num, self.time_limit(params))

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / self.time_limit(params)), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def render(self, params: EnvParams, timestep: TimeStep) -> np.ndarray | str:
        if params.render_mode == "rgb_array":
            return rgb_render(timestep.state.grid, timestep.state.agent, params.view_size)
        elif params.render_mode == "rich_text":
            return text_render(timestep.state.grid, timestep.state.agent)
        else:
            raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")
