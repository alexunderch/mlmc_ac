import optax
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import NamedTuple, Callable, Sequence, Any, Optional, NamedTuple

import optax._src.base as base
import optax._src.utils as utils
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

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray

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

def run_actorcritic_experiment_ppo(
    env_id: str,
    optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    max_t: int,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    vf_coeff: float,
    ent_coeff: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool
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
        action_logits, value = policy_state.apply_fn({'params': policy_state.params}, observations)
        pi = Categorical(logits=action_logits)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value
    
    @jax.jit
    def value(policy_state, observations):
        _, value = policy_state.apply_fn({'params': policy_state.params}, observations)
        return value
    
    def step_fn(carry, _):
        state, env_state, env_params, step_key, policy_state, ep_len =  carry
        env_step_key, act_key, key = jax.random.split(step_key, 3)
        action, log_prob, value = act(policy_state, state, act_key)
        next_state, new_env_state, reward, done, info = jit_step(env_step_key, env_state, action, env_params)
        return (
            (next_state, new_env_state, env_params, key, policy_state, ep_len+(1-done.astype(jnp.int32))), 
            Transition(state, action, log_prob, reward, value, done, info)
        )

    class ActorCritic(nn.Module):
        num_actions: jnp.ndarray

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            emb = nn.Sequential([
                nn.Dense(128),
                nn.tanh,
            ])(x)

            policy_logits = nn.Sequential([
                nn.Dense(128),
                nn.tanh,
                nn.Dense(features=self.num_actions)
            ])(emb)

            values = nn.Sequential([
                nn.Dense(128),
                nn.tanh,
                nn.Dense(1)
            ])(emb)

            return policy_logits, values.squeeze(-1)

    network = ActorCritic(env.num_actions)        

    init_network_key, reset_key, key = jax.random.split(jax.random.key(seed), 3)
    initial_obs, env_state = env.reset(reset_key, env_params)

    policy_state = TrainState.create(
        apply_fn=jax.jit(network.apply),
        params=network.init(init_network_key, initial_obs)['params'],
        tx=optimiser
    )

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + gamma * next_value * (1 - done) - value
            gae = (
                delta
                + gamma * gae_lambda * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    
    def ppo_update(policy_state, env_state, env_params, traj_batch, advantages, targets, rng):
        def _loss_fn(params, traj_batch, gae, targets):
            # RERUN NETWORK
            action_logits, value = network.apply({'params': params}, traj_batch.observation)
            pi = Categorical(logits=action_logits)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-clip_eps, clip_eps)
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = (
                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )
                * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            total_loss = (
                loss_actor
                + vf_coeff * value_loss
                - ent_coeff * entropy
            )
            return total_loss, (value_loss, loss_actor, entropy)

        loss_grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

        if mlmc_correction:
            #truncated geometric distribution
            rng, _rng = jax.random.split(rng)
            N = 6
            p = 0.5
            #https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(jnp.floor(jnp.log(1-jax.random.uniform(rng)*(1-(1-p)**N))/jnp.log(1-p)).item())
            
            (loss_value, (value_loss, loss_actor, entropy)), grads = loss_grad_fn(
                policy_state.params,
                jax.tree.map(lambda x: x[:batchsize_bound], traj_batch), advantages[:batchsize_bound], targets[:batchsize_bound]
            ) #0
            
            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(jnp.ceil((2**(J-1)) * batchsize_bound))
            observation = traj_batch.observation[batchsize_bound]
            #mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                (observation, env_state, env_params, rng, policy_state, ep_len), traj_batch = jax.lax.scan(
                    step_fn, (observation, env_state, env_params, _rng, policy_state, jnp.array(0)), xs=None, length=J_t
                )
                
                last_val = value(policy_state, observation)
                advantages, targets = _calculate_gae(traj_batch, last_val)

                _, mlmc_grads_t = loss_grad_fn(policy_state.params, traj_batch, advantages, targets) #t
                _, mlmc_grads_tm1 = loss_grad_fn(policy_state.params, jax.tree.map(lambda x: x[:J_tm1], traj_batch), advantages[:J_tm1], targets[:J_tm1]) #tm1
                grads = jax.tree.map(lambda g, x, y: g + (2**J) * (x - y), grads, mlmc_grads_t, mlmc_grads_tm1)
        else:
            observation = traj_batch.observation[-1]

            (loss_value, (value_loss, loss_actor, entropy)), grads = loss_grad_fn(
                policy_state.params,
                traj_batch, advantages, targets
            )

        policy_state = policy_state.apply_gradients(grads=grads)
        return loss_value, policy_state, (value_loss, loss_actor, entropy), observation, env_state

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, n_training_episodes+1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)
        initial_env_state = env_state
        initial_state = state

        (state, env_state, env_params, key, policy_state, ep_len), traj_batch = jax.lax.scan(
            step_fn, (initial_state, initial_env_state, env_params, step_key, policy_state, jnp.array(0)), xs=None, length=max_t
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)

        last_val = value(policy_state, initial_state)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        loss, policy_state, (value_loss, actor_loss, entropy), state, env_state = ppo_update(
            policy_state,
            initial_env_state,
            env_params,
            traj_batch,
            advantages,
            targets,
            loss_key,
        )

        wandb.log(
            {
                "Loss": loss.mean().item(),
                "VLoss": value_loss.mean().item(),
                "ActorLoss": actor_loss.mean().item(),
                "Entropy": entropy.mean().item(),
                "Step": i_episode,
                "Average_reward": np.mean(scores_deque),
                "Episode_length": np.mean(lengths_deque)
            }
        )
            
@hydra.main(config_path=".", config_name="config_ppo.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:

    dict_config = wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
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

    opt = optax.chain(
        optax.clip_by_global_norm(1000.),
        optimisers[dict_config["optimiser"]]
    )
    run_actorcritic_experiment_ppo(
        optimiser=opt, 
        **dict_config["experiment"], 
        seed=dict_config["seed"]
    )


if __name__ == "__main__":
   main()