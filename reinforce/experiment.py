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
from distrax import Categorical
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState
from optax import scale_by_learning_rate
from optax._src.numerics import abs_sq as _abs_sq
from optax.tree_utils import tree_add

warnings.filterwarnings("ignore")

import gymnax
import gymnax.wrappers
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


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
    n_training_episodes: int,
    gamma: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool = False,
):
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    env, env_params = gymnax.make(env_id)
    env = gymnax.wrappers.LogWrapper(env)

    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    max_t = env_params.max_steps_in_episode

    @jax.jit
    def act(policy_state, observations, key):
        return Categorical(
            logits=policy_state.apply_fn({"params": policy_state.params}, observations)
        ).sample(seed=key)

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

    init_key, reset_key, key = jax.random.split(jax.random.key(seed), 3)
    initial_obs, env_state = env.reset(reset_key, env_params)
    batchsize_bound = int(batchsize_bound)

    policy = nn.Sequential([nn.Dense(features=env.num_actions)])

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=policy.init(init_key, initial_obs)["params"],
        tx=optimiser,
    )

    def _calculate_returns(rewards, dones):
        def _one_step_return(carry, data):
            ret = carry
            reward, done = data
            carry = reward + gamma * ret * (1 - done)
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
    def loss_fn(params, observations, actions, returns):
        apply_fn = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))
        log_probs = Categorical(
            logits=apply_fn({"params": params}, observations)
        ).log_prob(actions)
        return -jnp.sum(log_probs * jax.lax.stop_gradient(returns))

    def reinforce_update(
        policy_state,
        env_state,
        env_params,
        observations,
        actions,
        rewards,
        dones,
        loss_rng,
        env_rng,
    ):

        loss_grad_fn = jax.value_and_grad(loss_fn)
        returns = _calculate_returns(rewards, dones)

        if mlmc_correction:
            # truncated geometric distribution
            rng, _rng = jax.random.split(loss_rng)
            N = 5
            p = 0.5
            # https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(
                jnp.floor(
                    jnp.log(1 - jax.random.uniform(rng) * (1 - (1 - p) ** N))
                    / jnp.log(1 - p)
                ).item()
            )

            loss_value, grads = loss_grad_fn(
                policy_state.params,
                observations[:batchsize_bound],
                actions[:batchsize_bound],
                returns[:batchsize_bound],
            )  # 0

            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(
                jnp.ceil((2 ** (J - 1)) * batchsize_bound)
            )
            observation = observations[batchsize_bound - 1]

            # mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                (observation, env_state, env_params, rng, policy_state, _), (
                    observations,
                    actions,
                    rewards,
                    dones,
                    _,
                ) = jax.lax.scan(
                    step_fn,
                    (
                        observations[0],
                        env_state,
                        env_params,
                        env_rng,
                        policy_state,
                        jnp.array(0),
                    ),
                    xs=None,
                    length=J_t,
                )

                returns = _calculate_returns(rewards, dones)

                _, mlmc_grads_t = loss_grad_fn(
                    policy_state.params, observations, actions, returns
                )  # t
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
        else:
            observation = observations[-1]
            loss_value, grads = loss_grad_fn(
                policy_state.params, observations, actions, returns
            )  # 0

        policy_state = policy_state.apply_gradients(grads=grads)
        grad_norm = optax.global_norm(grads)
        return loss_value, grad_norm, policy_state, observation, env_state

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, int(n_training_episodes) + 1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)

        (_state, _env_state, env_params, key, policy_state, ep_len), (
            states,
            actions,
            rewards,
            dones,
            infos,
        ) = jax.lax.scan(
            step_fn,
            (state, env_state, env_params, step_key, policy_state, jnp.array(0)),
            xs=None,
            length=jnp.minimum(max_t, batchsize_bound),
        )

        loss, grad_norm, policy_state, state, env_state = reinforce_update(
            policy_state,
            env_state,
            env_params,
            states,
            actions,
            rewards,
            dones,
            loss_key,
            step_key,
        )

        if not mlmc_correction:
            state = _state
            env_state = _env_state

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)

        wandb.log(
            {
                "REINFORCE_Loss": loss.mean().item(),
                "Grad_Norm": grad_norm.mean().item(),
                "Step": i_episode,
                "Average_Return": np.mean(scores_deque),
                "Episode_Length": np.mean(lengths_deque),
            }
        )


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    dict_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
    )

    # cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))
    wandb.config = dict_config

    optimisers = {
        "sgd": optax.sgd(learning_rate=dict_config["learning_rate"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["learning_rate"]),
        "adam": optax.adam(learning_rate=dict_config["learning_rate"]),
        "adamW": optax.adamw(learning_rate=dict_config["learning_rate"]),
    }

    run_reinforce_experiment(
        optimiser=optimisers[dict_config["optimiser"]],
        **dict_config["experiment"],
        seed=dict_config["seed"]
    )


if __name__ == "__main__":
    main()
