import math
import warnings
from collections import deque
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import optax._src.base as base
import optax._src.utils as utils
import pandas as pd
import yaml
from distrax import Categorical
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState
from pogema.envs import _make_pogema
from pogema.grid_config import GridConfig
from pogema.integrations.sample_factory import AutoResetWrapper
from pogema_toolbox.create_env import MultiMapWrapper
from pogema_toolbox.registry import ToolboxRegistry

warnings.filterwarnings("ignore")

import gymnax
import gymnax.wrappers
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

DEBUG_POGEMA = False


def isr_decay(initial_value: float) -> base.Schedule:
    """Constructs a square root decaying schedule.

    Args:
      initial_value: value to decay from.

    Returns:
      schedule
        A function that maps step counts to values.
    """
    return lambda count: initial_value / jnp.sqrt(count + 1.0)


class AcceleratedTraceState(NamedTuple):
    """State for the Adam algorithm."""

    count: jax.Array  # shape=(), dtype=jnp.int32.
    trace_f: base.Params


def accelerated_trace(
    learning_rate: float,
    decay_theta: float,
    decay_eta: float,
    decay_beta: float,
    decay_p: float,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:

    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        return AcceleratedTraceState(
            count=jnp.zeros((), dtype=jnp.int32), trace_f=params
        )

    def update_fn(updates, state, params):
        count = state.count
        mul_by_alpha = lambda alpha, x: jax.tree_util.tree_map(lambda y: alpha * y, x)

        trace_g = jax.tree_util.tree_map(
            lambda x, x_f: mul_by_alpha(decay_theta, x_f)
            + mul_by_alpha(1.0 - decay_theta, x),
            params,
            state.trace_f,
        )

        trace_f_update_fn = lambda g, t: t - decay_p * learning_rate(count) * g
        new_trace_f = jax.tree_util.tree_map(trace_f_update_fn, updates, trace_g)

        updates = jax.tree_util.tree_map(
            lambda x, x_f, next_x_f, x_g: (
                +mul_by_alpha(decay_eta, next_x_f)
                + mul_by_alpha(decay_p - decay_eta, x_f)
                + mul_by_alpha((1.0 - decay_p) * (1.0 - decay_beta) - 1.0, x)
                + mul_by_alpha((1.0 - decay_p) * decay_beta, x_g)
            ),
            params,
            state.trace_f,
            new_trace_f,
            trace_g,
        )
        new_trace_f = utils.cast_tree(new_trace_f, accumulator_dtype)

        return updates, AcceleratedTraceState(trace_f=new_trace_f, count=count + 1)

    return base.GradientTransformation(init_fn, update_fn)


def run_reinforce_experiment(
    env_id: str,
    optimiser: optax.GradientTransformation,
    reward_optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool = False,
    total_samples: Optional[int] = None,
    env_kwargs: Optional[dict] = None,
):
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    metrics_deque = deque(maxlen=100)

    total_samples: int = (
        int(total_samples)
        if total_samples is not None
        else batchsize_bound * n_training_episodes
    )

    sample_counter: int = 0
    stopping_criterium = lambda k: k > total_samples

    if "pogema" not in env_id:
        metrics_key = "Nil"

        env, env_params = gymnax.make(env_id)
        env = gymnax.wrappers.LogWrapper(env)

        jit_step = jax.jit(env.step)
        jit_reset = jax.jit(env.reset)

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, ep_len = carry
            env_step_key, act_key, key = jax.random.split(step_key, 3)
            action = act(policy_state, state, act_key)
            next_state, new_env_state, reward, done, info = jit_step(
                env_step_key, env_state, action, env_params
            )
            return (
                (
                    next_state,
                    new_env_state,
                    env_params,
                    key,
                    policy_state,
                    ep_len + (1 - done.astype(jnp.int32)),
                ),
                (state, action, reward, done, info),
            )

    else:

        map_filename = env_kwargs.get("map_filename")
        env_kwargs.pop("map_filename")
        if map_filename is None:
            config = GridConfig(**env_kwargs, seed=seed + 1, num_agents=1)
            env = _make_pogema(config)
            env = AutoResetWrapper(env)
        else:
            with open(map_filename, "r") as f:
                maps = yaml.safe_load(f)
            ToolboxRegistry.register_maps(maps)
            env = _make_pogema(
                GridConfig(
                    **env_kwargs,
                    seed=seed + 1,
                    num_agents=1,
                )
            )
            env = MultiMapWrapper(env)

        metrics_key = "avg_throughput" if config.on_target == "restart" else "ISR"

        if DEBUG_POGEMA:
            from pogema import AnimationMonitor

            env = AnimationMonitor(env)

        env_params = None

        @dataclass
        class LogEnvState:
            env_state: jax.Array
            episode_returns: jax.Array
            episode_lengths: jax.Array
            episode_metrics: jax.Array
            returned_episode_returns: jax.Array
            returned_episode_lengths: jax.Array
            returned_episode_metrics: jax.Array

        def callback_step(observation, key, env_state, env_params):
            action = act(policy_state, observation, key)
            observation, reward, terminated, truncated, info = env.step(action=[action])

            metrics = jnp.array(0.0)
            if "metrics" in info[0]:
                metrics = jnp.array(info[0]["metrics"][metrics_key])

            return (
                jnp.array(observation[0]).reshape(-1),
                action,
                (env_state + 1).astype(jnp.int32),
                jnp.array(reward[0]),
                jax.lax.cond(
                    metrics_key == "ISR",
                    lambda term, trunc: jnp.maximum(term, trunc).astype(jnp.int32),
                    lambda term, trunc: trunc.astype(jnp.int32),
                    jnp.array(terminated[0]),
                    jnp.array(truncated[0]),
                ),
                metrics,
            )

        @jax.jit
        def callback_reset(seed, env_params):
            int_seed = jax.random.randint(seed, (), 0, 1000)
            observation, _ = env.reset(seed=int_seed)
            return jnp.array(observation[0]).reshape(-1)

        def jit_reset(*args):
            reset_shape = jax.ShapeDtypeStruct((27,), jnp.float32)
            return jax.pure_callback(callback_reset, reset_shape, *args), LogEnvState(
                jnp.array(0), 0, 0, 0, 0, 0, 0
            )

        def jit_step(*args):

            step_shape = (
                jax.ShapeDtypeStruct((27,), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.float32),
            )
            return jax.experimental.io_callback(callback_step, step_shape, *args)

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, ep_len = carry
            act_key, key = jax.random.split(step_key, 2)
            next_state, action, new_env_state, reward, done, info = jit_step(
                state, act_key, env_state.env_state, env_params
            )

            new_episode_metrics = info
            new_episode_return = env_state.episode_returns + reward
            new_episode_length = env_state.episode_lengths + 1
            env_state = LogEnvState(
                env_state=new_env_state,
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                episode_metrics=new_episode_metrics * (1 - done),
                returned_episode_returns=env_state.returned_episode_returns * (1 - done)
                + new_episode_return * done,
                returned_episode_lengths=env_state.returned_episode_lengths * (1 - done)
                + new_episode_length * done,
                returned_episode_metrics=env_state.returned_episode_metrics * (1 - done)
                + new_episode_metrics * done,
            )

            return (
                (
                    next_state,
                    env_state,
                    env_params,
                    key,
                    policy_state,
                    ep_len + (1 - done.astype(jnp.int32)),
                ),
                (state, action, reward, done, info),
            )

    @jax.jit
    def act(policy_state, observations, key):
        return Categorical(
            logits=policy_state.apply_fn({"params": policy_state.params}, observations)
        ).sample(seed=key)

    policy = nn.Sequential(
        [
            nn.Dense(
                features=(
                    env.num_actions if "pogema" not in env_id else env.action_space.n
                )
            )
        ]
    )

    class RewardTracker(nn.Module):
        output_dim: int

        @nn.compact
        def __call__(self):
            return self.param(
                "eta", lambda rng, shape: jnp.zeros(shape), (self.output_dim,)
            )

    reward_tracker = RewardTracker(1)

    init_policy_key, init_reward_key, reset_key, key = jax.random.split(
        jax.random.key(seed), 4
    )
    initial_obs, env_state = jit_reset(reset_key, env_params)
    batchsize_bound = int(batchsize_bound)

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=policy.init(init_policy_key, initial_obs)["params"],
        tx=optimiser,
    )

    reward_tracker_state = TrainState.create(
        apply_fn=jax.jit(reward_tracker.apply),
        params=reward_tracker.init(init_reward_key)["params"],
        tx=reward_optimiser,
    )

    def _calculate_returns(rewards, dones, av_reward):
        def _one_step_return(carry, data):
            ret = carry
            reward, done = data
            carry = reward - av_reward.squeeze() + ret * (1 - done)
            return carry, carry

        _, returns = jax.lax.scan(
            _one_step_return,
            0,
            (rewards, dones),
            reverse=True,
        )
        eps = jnp.finfo(jnp.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    @jax.jit
    def policy_loss_fn(params, observations, actions, returns):
        apply_fn = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))
        log_probs = Categorical(
            logits=apply_fn({"params": params}, observations)
        ).log_prob(actions)
        return -jnp.sum(log_probs * jax.lax.stop_gradient(returns))

    @jax.jit
    def reward_tracker_loss_fn(params, returns):
        apply_fn = reward_tracker_state.apply_fn
        av_reward = apply_fn({"params": params})
        return (
            optax.l2_loss(jnp.tile(av_reward, returns.shape), returns).mean(),
            av_reward,
        )

    def reinforce_update(
        policy_state,
        reward_tracker_state,
        env_state,
        env_params,
        observation,
        rng,
        env_rng,
        sample_counter,
    ):

        loss_grad_fn = jax.value_and_grad(policy_loss_fn)
        reward_tracker_grad_fn = jax.value_and_grad(
            reward_tracker_loss_fn, has_aux=True
        )

        if mlmc_correction:
            # truncated geometric distribution
            rng, _rng = jax.random.split(rng)
            N = 5
            p = 0.5
            # https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(
                jnp.floor(
                    jnp.log(1 - jax.random.uniform(rng) * (1 - (1 - p) ** N))
                    / jnp.log(1 - p)
                ).item()
            )
            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(
                jnp.ceil((2 ** (J - 1)) * batchsize_bound)
            )

            (observation, env_state, env_params, key, policy_state, _), (
                observations,
                actions,
                rewards,
                dones,
                _,
            ) = jax.lax.scan(
                step_fn,
                (
                    observation,
                    env_state,
                    env_params,
                    env_rng,
                    policy_state,
                    jnp.array(0),
                ),
                xs=None,
                length=J_t,
                unroll=False,
            )

            (_, av_rewards), reward_tracker_grads = reward_tracker_grad_fn(
                reward_tracker_state.params, rewards[:batchsize_bound]
            )

            returns = _calculate_returns(
                rewards[:batchsize_bound], dones[:batchsize_bound], av_rewards
            )

            loss_value, grads = loss_grad_fn(
                policy_state.params,
                observations[:batchsize_bound],
                actions[:batchsize_bound],
                returns[:batchsize_bound],
            )  # 0

            # mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                sample_counter += J_t

                (_, av_rewards_t), reward_tracker_grads_t = reward_tracker_grad_fn(
                    reward_tracker_state.params, rewards
                )

                returns = _calculate_returns(rewards, dones, av_rewards_t)

                _, mlmc_grads_t = loss_grad_fn(
                    policy_state.params, observations, actions, returns
                )  # t

                (_, av_rewards_tm1), reward_tracker_grads_tm1 = reward_tracker_grad_fn(
                    reward_tracker_state.params, rewards[:J_tm1]
                )

                returns = _calculate_returns(
                    rewards[:J_tm1], dones[:J_tm1], av_rewards_tm1
                )

                _, mlmc_grads_tm1 = loss_grad_fn(
                    policy_state.params,
                    observations[:J_tm1],
                    actions[:J_tm1],
                    returns[:J_tm1],
                )  # tm1

                grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    grads,
                    mlmc_grads_t,
                    mlmc_grads_tm1,
                )

                reward_tracker_grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    reward_tracker_grads,
                    reward_tracker_grads_t,
                    reward_tracker_grads_tm1,
                )
            else:
                sample_counter += batchsize_bound
        else:
            sample_counter += batchsize_bound
            (observation, env_state, env_params, key, policy_state, _), (
                observations,
                actions,
                rewards,
                dones,
                _,
            ) = jax.lax.scan(
                step_fn,
                (
                    observation,
                    env_state,
                    env_params,
                    env_rng,
                    policy_state,
                    jnp.array(0),
                ),
                xs=None,
                length=batchsize_bound,
            )

            (_, av_rewards), reward_tracker_grads = reward_tracker_grad_fn(
                reward_tracker_state.params, rewards
            )

            returns = _calculate_returns(rewards, dones, av_rewards)

            loss_value, grads = loss_grad_fn(
                policy_state.params, observations, actions, returns
            )  # 0

        grad_norm = optax.global_norm(grads)

        policy_state = policy_state.apply_gradients(grads=grads)
        reward_tracker_state = reward_tracker_state.apply_gradients(
            grads=reward_tracker_grads
        )

        return (
            (loss_value, av_rewards),
            grad_norm,
            policy_state,
            reward_tracker_state,
            observation,
            env_state,
            key,
            sample_counter,
        )

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, int(n_training_episodes) + 1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)

        if stopping_criterium(sample_counter):
            break

        (
            loss,
            grad_norm,
            policy_state,
            reward_tracker_state,
            state,
            env_state,
            key,
            sample_counter,
        ) = reinforce_update(
            policy_state,
            reward_tracker_state,
            env_state,
            env_params,
            state,
            loss_key,
            step_key,
            sample_counter,
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)
        if "pogema" in env_id:
            metrics_deque.append(env_state.returned_episode_metrics)

        wandb.log(
            {
                "REINFORCE_Loss": loss[0].mean().item(),
                "Average_Reward_Tracker": loss[1].mean().item(),
                "Grad_Norm": grad_norm.mean().item(),
                "Step": i_episode,
                "Episode_Return": np.mean(scores_deque),
                "Episode_Length": np.mean(lengths_deque),
                f"Episode_{metrics_key}": np.mean(metrics_deque),
                "env_steps": sample_counter,
            },
        )


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    dict_config = wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
    )
    wandb.define_metric("env_steps")

    optimisers = {
        "sgd": optax.sgd(learning_rate=dict_config["learning_rate"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["learning_rate"]),
        "adam": optax.adam(learning_rate=dict_config["learning_rate"]),
        "adamW": optax.adamw(learning_rate=dict_config["learning_rate"]),
    }

    run_reinforce_experiment(
        optimiser=optimisers[dict_config["optimiser"]],
        reward_optimiser=optimisers[dict_config["optimiser"]],
        **dict_config["experiment"],
        seed=dict_config["seed"],
    )


if __name__ == "__main__":
    main()
