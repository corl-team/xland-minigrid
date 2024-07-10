from __future__ import annotations

from typing import Any

import jax

from .environment import Environment, EnvParamsT
from .types import EnvCarryT, IntOrArray, State, TimeStep


class Wrapper(Environment[EnvParamsT, EnvCarryT]):
    def __init__(self, env: Environment[EnvParamsT, EnvCarryT]):
        self._env = env

    # Question: what if wrapper adds new parameters to the dataclass?
    # Solution: do this after applying the wrapper:
    #   env_params = wrapped_env.default_params(**dataclasses.asdict(original_params))
    def default_params(self, **kwargs) -> EnvParamsT:
        return self._env.default_params(**kwargs)

    def num_actions(self, params: EnvParamsT) -> int:
        return self._env.num_actions(params)

    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        return self._env.observation_shape(params)

    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        return self._env._generate_problem(params, key)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._env.reset(params, key)

    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        return self._env.step(params, timestep, action)

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]):
        return self._env.render(params, timestep)


# gym and gymnasium style reset (on the same step with termination)
class GymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self._env.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# dm_env and envpool style reset (on the next step after termination)
class DmEnvAutoResetWrapper(Wrapper):
    def step(self, params, timestep, action):
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self._env.reset(params, timestep.state.key),
            lambda: self._env.step(params, timestep, action),
        )
        return timestep


class DirectionObsWrapper(Wrapper):
    def observation_shape(self, params):
        # base_shape = self._env.observation_shape(params)
        base_shape = {
            "img": self._env.observation_shape(params),
            "direction": 4,
        }
        # if not isinstance(base_shape, dict):
        #     assert isinstance(base_shape, tuple)
        #     assert len(base_shape) == 3
        #     base_shape = {
        #         "img": base_shape,
        #     }
        #
        # assert "img" in base_shape
        # assert not isinstance(base_shape, tuple)
        # base_shape.update(direction=4)

        return base_shape

    def __extend_obs(self, timestep):
        extended_obs = {
            "img": timestep.observation,
            "direction": jax.nn.one_hot(timestep.state.agent.direction, num_classes=4),
        }
        timestep = timestep.replace(observation=extended_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep


# class ExtendedObsWrapper(Wrapper):
#     # TODO: fix typing for the observation shape
#     def observation_shape(self, params):
#         return {
#             "img": self._env.observation_shape(params),
#             "direction": 4,
#             "goal_encoding": params.state.goal_encoding.shape,
#             "rule_encoding": params.state.rule_encoding.shape,
#         }
#
#     def __extend_obs(self, timestep):
#         extended_obs = {
#             "img": timestep.observation,
#             "direction": jax.nn.one_hot(timestep.state.agent.direction, num_classes=4),
#             "goal_encoding": timestep.state.goal_encoding,
#             "rule_encoding": timestep.state.rule_encoding,
#         }
#         timestep = timestep.replace(observation=extended_obs)
#         return timestep
#
#     def reset(self, params, key):
#         timestep = self._env.reset(params, key)
#         timestep = self.__extend_obs(timestep)
#         return timestep
#
#     def step(self, params, timestep, action):
#         timestep = self._env.step(params, timestep, action)
#         timestep = self.__extend_obs(timestep)
#         return timestep
