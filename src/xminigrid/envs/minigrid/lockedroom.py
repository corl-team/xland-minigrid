import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import horizontal_line, room, sample_direction, vertical_line
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]

_wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]

_allowed_colors = jnp.array(
    (
        Colors.YELLOW,
        Colors.PURPLE,
        Colors.GREEN,
        Colors.GREY,
        Colors.BLUE,
        Colors.RED,
    )
)


class LockedRoom(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=19, width=19)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 10 * params.height

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        key, rooms_key, colors_key, objects_key, coords_key, agent_pos_key, agent_dir_key = jax.random.split(key, num=7)

        # set up rooms
        grid = room(params.height, params.width)
        grid = vertical_line(grid, params.width // 2 - 2, 0, params.height, _wall_tile)
        grid = vertical_line(grid, params.width // 2 + 2, 0, params.height, _wall_tile)

        for i in range(1, 3):
            grid = horizontal_line(grid, 0, i * (params.height // 3), params.width // 2 - 2, _wall_tile)
            grid = horizontal_line(
                grid, params.width // 2 + 2, i * (params.height // 3), params.width // 2 - 2, _wall_tile
            )

        # hardcoded, but easier and faster to sample
        room_corners = jnp.array(
            (
                (0, 0),
                (params.height // 3, 0),
                (2 * (params.height // 3), 0),
                (0, params.width // 2 + 2),
                (params.height // 3, params.width // 2 + 2),
                (2 * (params.height // 3), params.width // 2 + 2),
            )
        )
        doors_idxs = jnp.array(
            (
                # left doors
                (params.height // 2 - (params.height // 3), params.width // 2 - 2),
                (params.height // 2, params.width // 2 - 2),
                (params.height // 2 + (params.height // 3), params.width // 2 - 2),
                # right doors
                (params.height // 2 - (params.height // 3), params.width // 2 + 2),
                (params.height // 2, params.width // 2 + 2),
                (params.height // 2 + (params.height // 3), params.width // 2 + 2),
            )
        )
        colors_order = jax.random.permutation(colors_key, _allowed_colors)

        for i in range(6):
            grid = grid.at[doors_idxs[i][0], doors_idxs[i][1]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors_order[i]])

        goal_idx, key_idx = jax.random.choice(rooms_key, 6, shape=(2,), replace=False)
        coords = jax.random.randint(
            coords_key,
            shape=(2, 2),
            minval=jnp.array((1, 1)),
            maxval=jnp.array((params.height // 3, params.width // 2 - 2)),
        )

        # set up goal room and locked door (I know, it's a lot of indexing)
        target_color = grid[doors_idxs[goal_idx][0], doors_idxs[goal_idx][1], 1]
        grid = grid.at[doors_idxs[goal_idx][0], doors_idxs[goal_idx][1]].set(
            TILES_REGISTRY[Tiles.DOOR_LOCKED, target_color]
        )
        grid = grid.at[room_corners[goal_idx][0] + coords[0][0], room_corners[goal_idx][1] + coords[0][1]].set(
            TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]
        )
        # place key
        grid = grid.at[room_corners[key_idx][0] + coords[1][0], room_corners[key_idx][1] + coords[1][1]].set(
            TILES_REGISTRY[Tiles.KEY, target_color]
        )

        # sample agent position
        position = jax.random.randint(
            agent_pos_key,
            shape=(2,),
            minval=jnp.array((1, params.width // 2 - 1)),
            maxval=jnp.array((params.height - 1, params.width // 2 + 1)),
        )
        agent = AgentState(position=position, direction=sample_direction(agent_dir_key))

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
