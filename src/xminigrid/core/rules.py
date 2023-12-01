import abc

import jax
import jax.numpy as jnp
from flax import struct

from .constants import TILES_REGISTRY, Colors, Tiles
from .grid import equal, get_neighbouring_tiles, pad_along_axis

MAX_RULE_ENCODING_LEN = 6 + 1  # for idx


def check_rule(encodings, grid, agent, action, position):
    def _check(carry, encoding):
        grid, agent = carry
        # What if use lax.cond here instead? Will it be faster?
        grid, agent = jax.lax.switch(
            encoding[0],
            (
                # empty rule first, we use them as paddings during benchmark generation
                lambda: EmptyRule.decode(encoding)(grid, agent, action, position),
                lambda: AgentHoldRule.decode(encoding)(grid, agent, action, position),
                lambda: AgentNearRule.decode(encoding)(grid, agent, action, position),
                lambda: TileNearRule.decode(encoding)(grid, agent, action, position),
            ),
        )
        return (grid, agent), None

    (grid, agent), _ = jax.lax.scan(_check, (grid, agent), encodings)

    return grid, agent


class BaseRule(struct.PyTreeNode):
    @abc.abstractmethod
    def __call__(self, grid, agent, action, position):
        ...

    @classmethod
    @abc.abstractmethod
    def decode(cls, encoding):
        ...

    @abc.abstractmethod
    def encode(self):
        ...


# TODO: checks on only some actions can produce bugs, but it is faster. If it's going to cause trouble, reconsider.
class EmptyRule(BaseRule):
    def __call__(self, grid, agent, action, position):
        return grid, agent

    @classmethod
    def decode(cls, encoding):
        return cls()

    def encode(self):
        return jnp.zeros(MAX_RULE_ENCODING_LEN, dtype=jnp.uint8)


class AgentHoldRule(BaseRule):
    tile: jax.Array
    prod_tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _rule_fn(agent):
            agent = jax.lax.cond(
                equal(agent.pocket, self.tile),
                lambda: agent.replace(pocket=self.prod_tile),
                lambda: agent,
            )
            return agent

        # TODO: seems like this check is slowing down, 25k -> 19k FPS
        # agent = _rule_fn(agent)
        agent = jax.lax.cond(
            jnp.equal(action, 3),
            lambda: _rule_fn(agent),
            lambda: agent,
        )
        return grid, agent

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3], prod_tile=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(1), self.tile, self.prod_tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_RULE_ENCODING_LEN)


class AgentNearRule(BaseRule):
    tile: jax.Array
    prod_tile: jax.Array

    def __call__(self, grid, agent, action, position):
        def _rule_fn(grid):
            y, x = agent.position
            up, right, down, left = get_neighbouring_tiles(grid, y, x)

            grid = jax.lax.select(equal(up, self.tile), grid.at[y - 1, x].set(self.prod_tile), grid)
            grid = jax.lax.select(equal(right, self.tile), grid.at[y, x + 1].set(self.prod_tile), grid)
            grid = jax.lax.select(equal(down, self.tile), grid.at[y + 1, x].set(self.prod_tile), grid)
            grid = jax.lax.select(equal(left, self.tile), grid.at[y, x - 1].set(self.prod_tile), grid)
            return grid

        grid = jax.lax.cond(
            jnp.equal(action, 0),
            lambda: _rule_fn(grid),
            lambda: grid,
        )
        return grid, agent

    @classmethod
    def decode(cls, encoding):
        return cls(tile=encoding[1:3], prod_tile=encoding[3:5])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(2), self.tile, self.prod_tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_RULE_ENCODING_LEN)


class TileNearRule(BaseRule):
    tile_a: jax.Array
    tile_b: jax.Array
    prod_tile: jax.Array

    def __call__(self, grid, agent, action, position):
        tile = grid[position[0], position[1]]
        tile_a = self.tile_a
        tile_b = self.tile_b

        def _rule_fn(grid):
            floor_tile = TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]
            y, x = position
            up, right, down, left = get_neighbouring_tiles(grid, y, x)

            grid = jax.lax.select(
                (equal(tile, tile_a) & equal(up, tile_b)) | (equal(tile, tile_b) & equal(up, tile_a)),
                grid.at[y - 1, x].set(self.prod_tile).at[y, x].set(floor_tile),
                grid,
            )
            grid = jax.lax.select(
                (equal(tile, tile_a) & equal(right, tile_b)) | (equal(tile, tile_b) & equal(right, tile_a)),
                grid.at[y, x + 1].set(self.prod_tile).at[y, x].set(floor_tile),
                grid,
            )
            grid = jax.lax.select(
                (equal(tile, tile_a) & equal(down, tile_b)) | (equal(tile, tile_b) & equal(down, tile_a)),
                grid.at[y + 1, x].set(self.prod_tile).at[y, x].set(floor_tile),
                grid,
            )
            grid = jax.lax.select(
                (equal(tile, tile_a) & equal(left, tile_b)) | (equal(tile, tile_b) & equal(left, tile_a)),
                grid.at[y, x - 1].set(self.prod_tile).at[y, x].set(floor_tile),
                grid,
            )
            return grid

        grid = jax.lax.cond(
            jnp.equal(action, 4) & (equal(tile, self.tile_a) | equal(tile, self.tile_b)),
            lambda: _rule_fn(grid),
            lambda: grid,
        )
        return grid, agent

    @classmethod
    def decode(cls, encoding):
        return cls(tile_a=encoding[1:3], tile_b=encoding[3:5], prod_tile=encoding[5:7])

    def encode(self):
        encoding = jnp.hstack([jnp.asarray(3), self.tile_a, self.tile_b, self.prod_tile], dtype=jnp.uint8)
        return pad_along_axis(encoding, MAX_RULE_ENCODING_LEN)
