import optax
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import NamedTuple, Callable, Sequence, Any, Optional

import optax._src.base as base
import optax._src.utils as utils
from optax._src.numerics import abs_sq as _abs_sq
from optax import scale_by_learning_rate
from optax.tree_utils import tree_add
import math
import matplotlib.pyplot as plt
import pandas as pd
from flax.training.train_state import TrainState
from distrax import Categorical
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import gymnax.wrappers
import gymnax

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

def isr_decay(
    initial_value: float
) -> base.Schedule:
  """Constructs a square root decaying schedule.

  Args:
    initial_value: value to decay from.

  Returns:
    schedule
      A function that maps step counts to values.
  """
  return lambda count: initial_value/jnp.sqrt(count + 1.)

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
            count = jnp.zeros((), dtype=jnp.int32),
            trace_f = params
        )

    def update_fn(updates, state, params):
        count = state.count
        mul_by_alpha = lambda alpha, x: jax.tree_util.tree_map(lambda y: alpha * y, x)

        trace_g = jax.tree_util.tree_map(
            lambda x, x_f: mul_by_alpha(decay_theta, x_f) + mul_by_alpha(1. - decay_theta, x),
            params, state.trace_f
        )

        trace_f_update_fn = lambda g, t: t - decay_p * learning_rate(count) * g
        new_trace_f = jax.tree_util.tree_map(trace_f_update_fn, updates, trace_g)

        updates = jax.tree_util.tree_map(
            lambda x, x_f, next_x_f, x_g: (
                    + mul_by_alpha(decay_eta, next_x_f)
                    + mul_by_alpha(decay_p - decay_eta, x_f)
                    + mul_by_alpha((1. - decay_p)*(1. - decay_beta) - 1., x)
                    + mul_by_alpha((1. - decay_p)*decay_beta, x_g)
                ),
            params, state.trace_f, new_trace_f, trace_g
        )
        new_trace_f = utils.cast_tree(new_trace_f, accumulator_dtype)

        return updates, AcceleratedTraceState(trace_f=new_trace_f, count=count+1)

    return base.GradientTransformation(init_fn, update_fn)

def run_actorcritic_experiment_montecarlo(
    env_id: str,
    actor_optimiser: optax.GradientTransformation,
    critic_optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    max_t: int,
    gamma: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction_actor: bool=False,
    mlmc_correction_critic: bool=False
):
    # env = make(env_id, max_episode_steps=max_t)
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    env, env_params = gymnax.make(env_id)
    env = gymnax.wrappers.LogWrapper(env)

    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    @jax.jit
    def act(policy_state, observations, key):
        return Categorical(logits=policy_state.apply_fn({'params': policy_state.params}, observations)).sample(seed=key)
    
    def step_fn(carry, _):
        state, env_state, env_params, step_key, policy_state, ep_len =  carry
        env_step_key, act_key, key = jax.random.split(step_key, 3)
        action = act(policy_state, state, act_key)
        next_state, new_env_state, reward, done, info = jit_step(env_step_key, env_state, action, env_params)
        # jax.debug.print("{x}\t{y}\n{z}", x=action, y=reward, z=new_env_state)
        return (
            (next_state, new_env_state, env_params, key, policy_state, ep_len+(1-done.astype(jnp.int32))), 
            (state, action, reward, done, info)
        )

    policy = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(128),
        nn.tanh,
        nn.Dense(features=env.num_actions)
    ])

    critic = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(128),
        nn.tanh,
        nn.Dense(1)
    ])

    init_actor_key, init_critic_key, reset_key, key = jax.random.split(jax.random.key(seed), 4)
    initial_obs, env_state = env.reset(reset_key, env_params)

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=policy.init(init_actor_key, initial_obs)['params'],
        tx=actor_optimiser
    )

    critic_state = TrainState.create(
        apply_fn=jax.jit(critic.apply),
        params=critic.init(init_critic_key, initial_obs)['params'],
        tx=critic_optimiser
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

    def ac_update(policy_state, critic_state, env_state, env_params, observations, actions, rewards, rng):
        @jax.jit
        def actor_loss_fn(params, observations, actions, td_errors):
            apply_fn = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))
            action_distribution = Categorical(logits=apply_fn({"params": params}, observations))
            return -jnp.mean(action_distribution.log_prob(actions) * td_errors)

        @jax.jit
        def critic_loss_fn(params, returns, observations):
            apply_fn = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))
            values = apply_fn({"params": params}, observations).squeeze(-1)

            td_error = returns - values
            return optax.l2_loss(values, returns).mean(), td_error

        actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn)
        critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)

        if mlmc_correction_actor or mlmc_correction_critic:
            #truncated geometric distribution
            rng, _rng = jax.random.split(rng)
            N = 5
            p = 0.5
            #https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(jnp.floor(jnp.log(1-jax.random.uniform(rng)*(1-(1-p)**N))/jnp.log(1-p)).item())
            (critic_loss_value, td_errors), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                rewards[:batchsize_bound],
                observations[:batchsize_bound],
            )
            actor_loss_value, policy_grads = actor_loss_grad_fn(
                policy_state.params,
                observations[:batchsize_bound],
                actions[:batchsize_bound],
                td_errors[:batchsize_bound]
            )
            
            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(jnp.ceil((2**(J-1)) * batchsize_bound))
            observation = observations[batchsize_bound]

            #mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                (observation, env_state, env_params, rng, policy_state, ep_len), (observations, actions, rewards, dones, _) = jax.lax.scan(
                    step_fn, (observations[0], env_state, env_params, _rng, policy_state, jnp.array(0)), xs=None, length=J_t
                )

                returns = _calculate_returns(rewards, dones)

                if mlmc_correction_critic:
                    (_, td_errors), critic_mlmc_grads_t = critic_loss_grad_fn(critic_state.params, returns, observations) #t
                    _, critic_mlmc_grads_tm1 = critic_loss_grad_fn(critic_state.params, returns[:J_tm1], observations[:J_tm1]) #tm1
                    critic_grads = jax.tree.map(lambda g, x, y: g + (2**J) * (x - y), critic_grads, critic_mlmc_grads_t, critic_mlmc_grads_tm1)

                if mlmc_correction_actor:
                    if not mlmc_correction_critic:
                        _, td_errors = critic_loss_fn(critic_state.params, returns, observations)
                    _, policy_mlmc_grads_t = actor_loss_grad_fn(policy_state.params, observations, actions, td_errors) #t
                    _, policy_mlmc_grads_tm1 = actor_loss_grad_fn(policy_state.params, observations[:J_tm1], actions[:J_tm1], td_errors[:J_tm1]) #tm1
                    policy_grads = jax.tree.map(lambda g, x, y: g + (2**J) * (x - y), policy_grads, policy_mlmc_grads_t, policy_mlmc_grads_tm1)
        else:
            observation = observations[-1]
            (critic_loss_value, td_errors), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                rewards,
                observations
            )
            actor_loss_value, policy_grads = actor_loss_grad_fn(
                policy_state.params,
                observations,
                actions,
                td_errors
            )

        policy_state = policy_state.apply_gradients(grads=policy_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        loss_value = critic_loss_value + actor_loss_value
        return loss_value, policy_state, critic_state, observation, env_state

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, n_training_episodes+1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)
        initial_env_state = env_state
        initial_state = state

        (state, env_state, env_params, key, policy_state, ep_len), (states, actions, rewards, dones, infos) = jax.lax.scan(
            step_fn, (initial_state, initial_env_state, env_params, step_key, policy_state, jnp.array(0)), xs=None, length=max_t
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)
        returns = _calculate_returns(rewards, dones)

        loss, policy_state, critic_state, state, env_state = ac_update(
            policy_state,
            critic_state,
            initial_env_state,
            env_params,
            states,
            actions,
            returns,
            loss_key,
        )

        wandb.log(
            {
                "Loss": loss.mean().item(),
                "Step": i_episode,
                "Average_reward": np.mean(scores_deque),
                "Episode_length": np.mean(lengths_deque)
            }
        )

def run_actorcritic_experiment_td0(
    env_id: str,
    actor_optimiser: optax.GradientTransformation,
    critic_optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    max_t: int,
    gamma: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction_actor: bool=False,
    mlmc_correction_critic: bool=False
):
    # env = make(env_id, max_episode_steps=max_t)
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    env, env_params = gymnax.make(env_id)
    env = gymnax.wrappers.LogWrapper(env)

    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    @jax.jit
    def act(policy_state, observations, key):
        return Categorical(logits=policy_state.apply_fn({'params': policy_state.params}, observations)).sample(seed=key)
    
    def step_fn(carry, _):
        state, env_state, env_params, step_key, policy_state, ep_len =  carry
        env_step_key, act_key, key = jax.random.split(step_key, 3)
        action = act(policy_state, state, act_key)
        next_state, new_env_state, reward, done, info = jit_step(env_step_key, env_state, action, env_params)
        # jax.debug.print("{x}\t{y}\n{z}", x=action, y=reward, z=new_env_state)
        return (
            (next_state, new_env_state, env_params, key, policy_state, ep_len+(1-done.astype(jnp.int32))), 
            (state, next_state, action, reward, done, info)
        )


    policy = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(128),
        nn.tanh,
        nn.Dense(features=env.num_actions)
    ])

    critic = nn.Sequential([
        nn.Dense(64),
        nn.tanh,
        nn.Dense(128),
        nn.tanh,
        nn.Dense(1)
    ])

    init_actor_key, init_critic_key, reset_key, key = jax.random.split(jax.random.key(seed), 4)
    initial_obs, env_state = env.reset(reset_key, env_params)

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=policy.init(init_actor_key, initial_obs)['params'],
        tx=actor_optimiser
    )

    critic_state = TrainState.create(
        apply_fn=jax.jit(critic.apply),
        params=critic.init(init_critic_key, initial_obs)['params'],
        tx=critic_optimiser
    )

    def ac_update(policy_state, critic_state, env_state, env_params, observations, next_observations, actions, rewards, dones, rng):
        @jax.jit
        def actor_loss_fn(params, observations, actions, td_errors):
            apply_fn = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))
            action_distribution = Categorical(logits=apply_fn({"params": params}, observations))
            return -jnp.mean(action_distribution.log_prob(actions) * td_errors)

        @jax.jit
        def critic_loss_fn(params, returns, observations, next_observations, dones):
            apply_fn = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))
            values = apply_fn({"params": params}, observations).squeeze(-1)
            next_values = apply_fn({"params": params}, next_observations).squeeze(-1)
            td_target = returns + gamma * (1.-dones) * jax.lax.stop_gradient(next_values) 
            return optax.l2_loss(values, td_target).mean(), td_target - values

        actor_loss_grad_fn = jax.value_and_grad(actor_loss_fn)
        critic_loss_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)

        if mlmc_correction_actor or mlmc_correction_critic:
            #truncated geometric distribution
            rng, _rng = jax.random.split(rng)
            N = 5
            p = 0.5
            #https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(jnp.floor(jnp.log(1-jax.random.uniform(rng)*(1-(1-p)**N))/jnp.log(1-p)).item())
            (critic_loss_value, td_errors), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                rewards,
                observations[:batchsize_bound],
                next_observations[:batchsize_bound],
                dones[:batchsize_bound]
            )
            actor_loss_value, policy_grads = actor_loss_grad_fn(
                policy_state.params,
                observations[:batchsize_bound],
                actions[:batchsize_bound],
                td_errors[:batchsize_bound]
            )
            
            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(jnp.ceil((2**(J-1)) * batchsize_bound))
            observation = observations[batchsize_bound]

            #mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                (observation, env_state, env_params, rng, policy_state, ep_len), (observations, next_observations, actions, rewards, dones, _) = jax.lax.scan(
                    step_fn, (observations[0], env_state, env_params, _rng, policy_state, jnp.array(0)), xs=None, length=J_t
                )
                returns = rewards
                if mlmc_correction_critic:
                    (_, td_errors), critic_mlmc_grads_t = critic_loss_grad_fn(critic_state.params, returns, observations, next_observations, dones) #t
                    _, critic_mlmc_grads_tm1 = critic_loss_grad_fn(critic_state.params, returns[:J_tm1], observations[:J_tm1],  next_observations[:J_tm1], dones[:J_tm1]) #tm1
                    critic_grads = jax.tree.map(lambda g, x, y: g + (2**J) * (x - y), critic_grads, critic_mlmc_grads_t, critic_mlmc_grads_tm1)

                if mlmc_correction_actor:
                    if not mlmc_correction_critic:
                        _, td_errors = critic_loss_fn(critic_state.params, returns, observations, next_observations, dones)
                    td_errors = td_errors - td_errors.mean() / (td_errors.std() + 1e-6)
                    _, policy_mlmc_grads_t = actor_loss_grad_fn(policy_state.params, observations, actions, td_errors) #t
                    _, policy_mlmc_grads_tm1 = actor_loss_grad_fn(policy_state.params, observations[:J_tm1], actions[:J_tm1], td_errors[:J_tm1]) #tm1
                    policy_grads = jax.tree.map(lambda g, x, y: g + (2**J) * (x - y), policy_grads, policy_mlmc_grads_t, policy_mlmc_grads_tm1)

        else:
            observation = observations[-1]
            (critic_loss_value, td_errors), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                rewards,
                observations,
                next_observations,
                dones
            )
            actor_loss_value, policy_grads = actor_loss_grad_fn(
                policy_state.params,
                observations,
                actions,
                td_errors
            )

        policy_state = policy_state.apply_gradients(grads=policy_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        loss_value = critic_loss_value + actor_loss_value
        return loss_value, policy_state, critic_state, observation, env_state

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, n_training_episodes+1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)
        initial_env_state = env_state
        initial_state = state

        (state, env_state, env_params, key, policy_state, ep_len), (states, next_states, actions, rewards, dones, infos) = jax.lax.scan(
            step_fn, (initial_state, initial_env_state, env_params, step_key, policy_state, jnp.array(0)), xs=None, length=max_t
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)
        returns = rewards

        loss, policy_state, critic_state, state, env_state = ac_update(
            policy_state,
            critic_state,
            initial_env_state,
            env_params,
            states,
            next_states,
            actions,
            returns,
            dones,
            loss_key,
        )

        wandb.log(
            {
                "Loss": loss.mean().item(),
                "Step": i_episode,
                "Average_reward": np.mean(scores_deque),
                "Episode_length": np.mean(lengths_deque)
            }
        )

            
@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
    )

    wandb.config = dict_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    # cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(wandb.config)))
    # wandb.config = dict(cfg)

    optimisers = {
        "sgd": optax.sgd(learning_rate=dict_config["learning_rate"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["learning_rate"]),
        "adam": optax.adam(learning_rate=dict_config["learning_rate"]),
        "adamW": optax.adamw(learning_rate=dict_config["learning_rate"]),
        "accelerated_sgd":  accelerated_trace(learning_rate=lambda t: 0.1, **dict_config["momentum"]),
        "accelerated_sgd_adagrad":  optax.chain(
            optax.scale_by_rss(),
            accelerated_trace(learning_rate=lambda t: 0.1, **dict_config["momentum"])
        )
    }

    exp = {
        "td0": run_actorcritic_experiment_td0,
        "mc": run_actorcritic_experiment_montecarlo
    }
    
    actor_optimiser = optax.chain(
        optax.clip_by_global_norm(100.),
        optimisers[dict_config["actor_optimiser"]]
    )
    critic_optimiser = optax.chain(
        optax.clip_by_global_norm(100.),
        optimisers[dict_config["critic_optimiser"]]
    )

    exp[dict_config["type"]](
        actor_optimiser=actor_optimiser, 
        critic_optimiser=critic_optimiser, 
        **dict_config["experiment"], 
        seed=dict_config["seed"]
    )


if __name__ == "__main__":
   main()