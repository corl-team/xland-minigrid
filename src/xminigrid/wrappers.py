from __future__ import annotations

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

    def time_limit(self, params: EnvParamsT) -> int:
        return self._env.time_limit(params)

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
