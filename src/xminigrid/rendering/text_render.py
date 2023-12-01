import jax
import jax.numpy as jnp

from ..core.constants import Colors, Tiles
from ..types import AgentState, RuleSet

COLOR_NAMES = {
    Colors.END_OF_MAP: "red",
    Colors.UNSEEN: "white",
    Colors.EMPTY: "white",
    Colors.RED: "red",
    Colors.GREEN: "green",
    Colors.BLUE: "blue",
    Colors.PURPLE: "purple",
    Colors.YELLOW: "yellow",
    Colors.GREY: "grey",
    Colors.BLACK: "black",
    Colors.ORANGE: "orange",
    Colors.WHITE: "white",
}

TILE_STR = {
    Tiles.END_OF_MAP: "!",
    Tiles.UNSEEN: "?",
    Tiles.EMPTY: " ",
    Tiles.FLOOR: ".",
    Tiles.WALL: "☰",
    Tiles.BALL: "⏺",
    Tiles.SQUARE: "▪",
    Tiles.PYRAMID: "▲",
    Tiles.GOAL: "■",
    Tiles.DOOR_LOCKED: "L",
    Tiles.DOOR_CLOSED: "C",
    Tiles.DOOR_OPEN: "O",
    Tiles.KEY: "K",
}

# for ruleset printing
RULE_TILE_STR = {
    Tiles.FLOOR: "floor",
    Tiles.BALL: "ball",
    Tiles.SQUARE: "square",
    Tiles.PYRAMID: "pyramid",
    Tiles.GOAL: "goal",
    Tiles.KEY: "key",
}

PLAYER_STR = {0: "^", 1: ">", 2: "V", 3: "<"}


def _wrap_with_color(string, color):
    return f"[bold {color}]{string}[/bold {color}]"


# WARN: will NOT work under jit and needed for debugging mainly.
def render(grid: jax.Array, agent: AgentState | None = None) -> str:
    string = ""

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            tile_id, tile_color = grid[y, x]
            tile_str = TILE_STR[tile_id.item()]
            tile_color = COLOR_NAMES[tile_color.item()]

            if agent is not None and jnp.all(agent.position == jnp.array((y, x))):
                tile_str = PLAYER_STR[agent.direction.item()]
                tile_color = COLOR_NAMES[Colors.RED]

            string += _wrap_with_color(tile_str, tile_color)

        if y < grid.shape[0] - 1:
            string += "\n"

    return string


# WARN: This is also for debugging mainly!
def _text_encode_tile(tile):
    return f"{COLOR_NAMES[tile[1]]} {RULE_TILE_STR[tile[0]]}"


def _text_encode_goal(goal):
    goal_id = goal[0]
    if goal_id == 1:
        return f"AgentHold({_text_encode_tile(goal[1:3])})"
    elif goal_id == 3:
        return f"AgentNear({_text_encode_tile(goal[1:3])})"
    elif goal_id == 4:
        tile_a = _text_encode_tile(goal[1:3])
        tile_b = _text_encode_tile(goal[3:5])
        return f"TileNear({tile_a}, {tile_b})"
    else:
        raise RuntimeError(f"Rendering: Unknown goal id: {goal_id}")


def _text_encode_rule(rule):
    rule_id = rule[0]
    if rule_id == 1:
        tile = _text_encode_tile(rule[1:3])
        prod_tile = _text_encode_tile(rule[3:5])
        return f"AgentHold({tile}) -> {prod_tile}"
    elif rule_id == 2:
        tile = _text_encode_tile(rule[1:3])
        prod_tile = _text_encode_tile(rule[3:5])
        return f"AgentNear({tile}) -> {prod_tile}"
    elif rule_id == 3:
        tile_a = _text_encode_tile(rule[1:3])
        tile_b = _text_encode_tile(rule[3:5])
        prod_tile = _text_encode_tile(rule[5:7])
        return f"TileNear({tile_a}, {tile_b}) -> {prod_tile}"
    else:
        raise RuntimeError(f"Rendering: Unknown rule id: {rule_id}")


def print_ruleset(ruleset: RuleSet):
    print("GOAL:")
    print(_text_encode_goal(ruleset.goal.tolist()))
    print()
    print("RULES:")
    for rule in ruleset.rules.tolist():
        if rule[0] != 0:
            print(_text_encode_rule(rule))
    print()
    print("INIT TILES:")
    for tile in ruleset.init_tiles.tolist():
        if tile[0] != 0:
            print(_text_encode_tile(tile))
