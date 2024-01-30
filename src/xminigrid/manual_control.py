import argparse

import jax
import numpy as np
import pygame
import pygame.freetype
from pygame.event import Event

import xminigrid
from xminigrid.wrappers import GymAutoResetWrapper

from .environment import Environment, EnvParams


class ManualControl:
    def __init__(self, env: Environment, env_params: EnvParams):
        self.env = env
        self.env_params = env_params

        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self._key = jax.random.PRNGKey(0)

        self.timestep = None

        self.render_size = None
        self.window = None
        self.clock = None
        self.screen_size = 640
        self.closed = False

    def render(self) -> None:
        assert self.timestep is not None

        img = self.env.render(self.env_params, self.timestep)
        # [h, w, c] -> [w, h, c]
        img = np.transpose(img, axes=(1, 0, 2))

        if self.render_size is None:
            self.render_size = img.shape[:2]

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("xland-minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        # offset = 32 if self.agent_pov else 64
        bg = pygame.Surface((int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset)))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        font_size = 16
        text = "TODO: game goal encoding"
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = int(bg.get_height() - font_size * 1.5)
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(10)
        pygame.display.flip()

    def start(self) -> None:
        self.reset()

        """Start the window display with blocking event loop"""
        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: int) -> None:
        self.timestep = self._step(self.env_params, self.timestep, action)
        print(
            "StepType: ",
            self.timestep.step_type,
            "Discount: ",
            self.timestep.discount,
            "Reward: ",
            self.timestep.reward,
        )
        self.render()

    def reset(self) -> None:
        print("Reset!")
        self._key, reset_key = jax.random.split(self._key)

        self.timestep = self._reset(self.env_params, reset_key)
        self.render()
        print(
            "StepType: ",
            self.timestep.step_type,
            "Discount: ",
            self.timestep.discount,
            "Reward: ",
            self.timestep.reward,
        )

    def key_handler(self, event: Event) -> None:
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.close()
            return

        key_to_action = {
            "up": 0,
            "right": 1,
            "left": 2,
            "tab": 3,
            "left shift": 4,
            "space": 5,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def close(self) -> None:
        if self.window:
            pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-Empty-5x5", choices=xminigrid.registered_environments())
    parser.add_argument("--benchmark-id", type=str, default="trivial-1m", choices=xminigrid.registered_benchmarks())
    parser.add_argument("--ruleset-id", type=int, default=0)

    args = parser.parse_args()
    env, env_params = xminigrid.make(args.env_id)
    env = GymAutoResetWrapper(env)

    if "XLand" in args.env_id:
        bench = xminigrid.load_benchmark(args.benchmark_id)
        env_params = env_params.replace(ruleset=bench.get_ruleset(args.ruleset_id))

    control = ManualControl(env=env, env_params=env_params)
    control.start()
