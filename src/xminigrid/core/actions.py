from __future__ import annotations

import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias

from ..types import AgentState, GridState, IntOrArray
from .constants import DIRECTIONS, TILES_REGISTRY, Colors, Tiles
from .grid import check_can_put, check_pickable, check_walkable, equal

ActionOutput: TypeAlias = tuple[GridState, AgentState, jax.Array]


def _move(position: jax.Array, direction: jax.Array) -> jax.Array:
    direction = jax.lax.dynamic_index_in_dim(DIRECTIONS, direction, keepdims=False)
    new_position = position + direction
    return new_position


def move_forward(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = jnp.clip(
        _move(agent.position, agent.direction),
        min=jnp.array((0, 0)),
        max=jnp.array((grid.shape[0] - 1, grid.shape[1] - 1)),  # H, W
    )
    position = jax.lax.select(
        check_walkable(grid, next_position),
        next_position,
        agent.position,
    )
    new_agent = agent.replace(position=position)

    return grid, new_agent, new_agent.position


def turn_clockwise(grid: GridState, agent: AgentState) -> ActionOutput:
    new_direction = (agent.direction + 1) % 4
    new_agent = agent.replace(direction=new_direction)
    return grid, new_agent, agent.position


def turn_counterclockwise(grid: GridState, agent: AgentState) -> ActionOutput:
    new_direction = (agent.direction - 1) % 4
    new_agent = agent.replace(direction=new_direction)
    return grid, new_agent, agent.position


def pick_up(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = _move(agent.position, agent.direction)

    is_pickable = check_pickable(grid, next_position)
    is_empty_pocket = jnp.equal(agent.pocket[0], Tiles.EMPTY)
    # pick up only if pocket is empty and entity is pickable
    new_grid, new_agent = jax.lax.cond(
        is_pickable & is_empty_pocket,
        lambda: (
            grid.at[next_position[0], next_position[1]].set(TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]),
            agent.replace(
                pocket=TILES_REGISTRY[
                    grid[next_position[0], next_position[1], 0],
                    grid[next_position[0], next_position[1], 1],
                ]
            ),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, next_position


def put_down(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = _move(agent.position, agent.direction)

    can_put = check_can_put(grid, next_position)
    not_empty_pocket = jnp.not_equal(agent.pocket[0], Tiles.EMPTY)
    new_grid, new_agent = jax.lax.cond(
        can_put & not_empty_pocket,
        lambda: (
            grid.at[next_position[0], next_position[1]].set(agent.pocket),
            agent.replace(pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, next_position


# TODO: may be this should be open_door action? toggle is too general and box is not supported yet
def toggle(grid: GridState, agent: AgentState) -> ActionOutput:
    next_position = _move(agent.position, agent.direction)
    next_tile = grid[next_position[0], next_position[1]]

    # check door_locked
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_LOCKED) & equal(agent.pocket, TILES_REGISTRY[Tiles.KEY, next_tile[1]]),
        grid.at[next_position[0], next_position[1]].set(TILES_REGISTRY[Tiles.DOOR_OPEN, next_tile[1]]),
        grid,
    )
    # check door_closed
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_CLOSED),
        grid.at[next_position[0], next_position[1]].set(TILES_REGISTRY[Tiles.DOOR_OPEN, next_tile[1]]),
        new_grid,
    )
    # check door_open
    new_grid = jax.lax.select(
        jnp.equal(next_tile[0], Tiles.DOOR_OPEN),
        grid.at[next_position[0], next_position[1]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, next_tile[1]]),
        new_grid,
    )
    return new_grid, agent, next_position


def take_action(grid: GridState, agent: AgentState, action: IntOrArray) -> ActionOutput:
    # This will evaluate all actions.
    # Can we fix this and choose only one function? It'll speed everything up dramatically.
    actions = (
        lambda: move_forward(grid, agent),
        lambda: turn_clockwise(grid, agent),
        lambda: turn_counterclockwise(grid, agent),
        lambda: pick_up(grid, agent),
        lambda: put_down(grid, agent),
        lambda: toggle(grid, agent),
    )
    new_grid, new_agent, changed_position = jax.lax.switch(action, actions)

    return new_grid, new_agent, changed_position
