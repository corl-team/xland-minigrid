# example on how to restore checkpoints and use them for inference
# TODO: add argparse arguments to load generic checkpoints and envs
# TODO: move this to examples/
import imageio
import jax
import jax.numpy as jnp
import orbax.checkpoint
import xminigrid
from nn import ActorCriticRNN
from xminigrid.rendering.text_render import print_ruleset
from xminigrid.wrappers import GymAutoResetWrapper

TOTAL_EPISODES = 10


def main():
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint = orbax_checkpointer.restore("../xland-minigrid-data/checkpoints")
    config = checkpoint["config"]
    params = checkpoint["params"]

    env, env_params = xminigrid.make("XLand-MiniGrid-R1-9x9")
    env = GymAutoResetWrapper(env)

    ruleset = xminigrid.load_benchmark("trivial-1m").get_ruleset(3)
    env_params = env_params.replace(ruleset=ruleset)

    model = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config["action_emb_dim"],
        rnn_hidden_dim=config["rnn_hidden_dim"],
        rnn_num_layers=config["rnn_num_layers"],
        head_hidden_dim=config["head_hidden_dim"],
    )
    # jitting all functions
    apply_fn, reset_fn, step_fn = jax.jit(model.apply), jax.jit(env.reset), jax.jit(env.step)

    # initial inputs
    prev_reward = jnp.asarray(0)
    prev_action = jnp.asarray(0)
    hidden = model.initialize_carry(1)

    # for logging
    total_reward, num_episodes = 0, 0
    rendered_imgs = []

    rng = jax.random.key(0)
    rng, _rng = jax.random.split(rng)

    timestep = reset_fn(env_params, _rng)
    rendered_imgs.append(env.render(env_params, timestep))
    while num_episodes < TOTAL_EPISODES:
        rng, _rng = jax.random.split(rng)
        dist, value, hidden = apply_fn(
            params,
            {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hidden,
        )
        action = dist.sample(seed=_rng).squeeze()

        timestep = step_fn(env_params, timestep, action)
        prev_action = action
        prev_reward = timestep.reward

        total_reward += timestep.reward.item()
        num_episodes += int(timestep.last().item())

        rendered_imgs.append(env.render(env_params, timestep))

    print("Total reward:", total_reward)
    print_ruleset(ruleset)
    imageio.mimsave("rollout.mp4", rendered_imgs, fps=8, format="mp4")
    # imageio.mimsave("rollout.gif", rendered_imgs, duration=1000 * 1 / 8, format="gif")


if __name__ == "__main__":
    main()
