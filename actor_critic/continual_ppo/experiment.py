import warnings
from collections import deque
from typing import Any, Callable, NamedTuple, Optional, Sequence

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
from pogema.envs import _make_pogema
from pogema.grid_config import GridConfig
from pogema.integrations.sample_factory import AutoResetWrapper

warnings.filterwarnings("ignore")

import gymnax
import gymnax.wrappers
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

DEBUG_POGEMA = False

class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray


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


def run_actorcritic_experiment_ppo(
    env_id: str,
    optimiser: optax.GradientTransformation,
    av_tracker_optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    max_t: int,
    gae_lambda: float,
    clip_eps: float,
    vf_coeff: float,
    av_vf_coeff: float,
    ent_coeff: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool,
    env_kwargs: Optional[dict] = None,
):
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    metrics_deque = deque(maxlen=100)

    @jax.jit
    def act(policy_state, observations, key):
        action_logits, value = policy_state.apply_fn(
            {"params": policy_state.params}, observations
        )
        pi = Categorical(logits=action_logits)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value

    @jax.jit
    def value(policy_state, observations):
        _, value = policy_state.apply_fn({"params": policy_state.params}, observations)
        return value
    
    if "pogema" not in env_id:
        env, env_params = gymnax.make(env_id)
        env = gymnax.wrappers.LogWrapper(env)

        jit_step = jax.jit(env.step)
        jit_reset = jax.jit(env.reset)

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, ep_len = carry
            env_step_key, act_key, key = jax.random.split(step_key, 3)
            action, log_prob, value = act(policy_state, state, act_key)
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
                Transition(state, action, log_prob, reward, value, done, info),
            )


    else:
        config = GridConfig(**env_kwargs, seed=seed + 1, num_agents=1)
        env = _make_pogema(config)
        env = AutoResetWrapper(env)

        metrics_key = "avg_throughput" if config.on_target == "restart" else "ISR"

        if DEBUG_POGEMA:
            import pogema.animation

            env = pogema.animation.AnimationMonitor(env)

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

        def callback_step(action, env_state, env_params):
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
                    jnp.array(terminated[0]), jnp.array(truncated[0]),
                ),
                metrics   
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
            action, log_prob, value = act(policy_state, state, act_key)

            next_state, action, new_env_state, reward, done, info = jit_step(
                action, env_state.env_state, env_params
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
                Transition(state, action, log_prob, reward, value, done, info),
            )

    class ActorCritic(nn.Module):
        num_actions: jnp.ndarray

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            policy_logits = nn.Sequential([nn.Dense(features=self.num_actions)])(x)
            values = nn.Sequential([nn.Dense(1), lambda x: jnp.squeeze(x, -1)])(x)

            return policy_logits, values
    
    class Tracker(nn.Module):
        output_dim: int

        @nn.compact
        def __call__(self):
            return (
                self.param(
                    "tracked_reward", lambda rng, shape: jnp.zeros(shape), (self.output_dim,)
                ),
                self.param(
                    "tracked_value", lambda rng, shape: jnp.zeros(shape), (self.output_dim,)
                ),
            )
    
    av_tracker = Tracker(1)

    network = ActorCritic(env.num_actions)

    init_network_key, init_reward_key, reset_key, key = jax.random.split(jax.random.key(seed), 4)
    initial_obs, env_state = env.reset(reset_key, env_params)

    policy_state = TrainState.create(
        apply_fn=jax.jit(network.apply),
        params=network.init(init_network_key, initial_obs)["params"],
        tx=optimiser,
    )

    tracker_state = TrainState.create(
        apply_fn=jax.jit(av_tracker.apply),
        params=av_tracker.init(init_reward_key)["params"],
        tx=av_tracker_optimiser,
    )

    def _calculate_gae(traj_batch, last_val, av_reward):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done.astype(jnp.float32),
                transition.value,
                transition.reward,
            )
            delta = reward - av_reward.astype(jnp.float32) + next_value * (1 - done) - value
            gae = delta + gae_lambda * (1 - done) * gae
            return (gae, value), gae
        last_val = last_val[0]
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def _loss_fn(params, traj_batch, gae, targets, av_value):
        # GRAIDENT OPTIMISATION PART
        # RERUN NETWORK
        action_logits, value = network.apply({"params": params}, traj_batch.observation)
        pi = Categorical(logits=action_logits)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value = value - av_value
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -clip_eps, clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

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
        
        total_loss = loss_actor + vf_coeff * value_loss - ent_coeff * entropy
        return total_loss, (value_loss, loss_actor, entropy)

    @jax.jit
    def _av_reward_tracker_loss_fn(params, returns, values):
        apply_fn = tracker_state.apply_fn
        av_reward, av_value = apply_fn({"params": params})
        av_reward_update = optax.l2_loss(av_reward.squeeze(), returns.mean())
        av_value_update = optax.l2_loss(av_value.squeeze(), values.mean())
        av_value *= av_vf_coeff
        total_loss = av_reward_update + av_value_update 
        return (
            total_loss,
            (av_reward, jnp.tile(av_value, values.shape)),
        )

    def ppo_update(
        policy_state,
        tracker_state,
        observation,
        env_state,
        env_params,
        rng,
        env_rng,
    ):

        loss_grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        reward_tracker_grad_fn = jax.value_and_grad(
            _av_reward_tracker_loss_fn, has_aux=True
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
            (observation, env_state, env_params, key, policy_state, _), traj_batch = jax.lax.scan(
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

            (_, (av_reward, av_value)), tracker_grads = (
                reward_tracker_grad_fn(tracker_state.params, traj_batch.reward, traj_batch.value)
            )

            
            last_val = value(policy_state, observation)
            advantages, targets = _calculate_gae(traj_batch, av_reward, last_val)

            (loss_value, (value_loss, loss_actor, entropy)), grads = loss_grad_fn(
                policy_state.params,
                jax.tree.map(lambda x: x[:batchsize_bound], traj_batch),
                advantages[:batchsize_bound],
                targets[:batchsize_bound],
                av_value[:batchsize_bound],
            )  # 0

            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound)), int(
                jnp.ceil((2 ** (J - 1)) * batchsize_bound)
            )
            observation = traj_batch.observation[batchsize_bound - 1]
            # mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                
                _, tracker_grads_t = reward_tracker_grad_fn(
                        tracker_state.params, traj_batch.reward, traj_batch.value
                )

                _, mlmc_grads_t = loss_grad_fn(
                    policy_state.params, traj_batch, advantages, targets, av_value
                )  # t

                _, tracker_grads_tm1 = reward_tracker_grad_fn(
                        tracker_state.params, traj_batch.reward[:J_tm1], traj_batch.value[:J_tm1]
                )

                _, mlmc_grads_tm1 = loss_grad_fn(
                    policy_state.params,
                    jax.tree.map(lambda x: x[:J_tm1], traj_batch),
                    advantages[:J_tm1],
                    targets[:J_tm1],
                    av_value[:J_tm1],
                )  # tm1
                
                grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    grads,
                    mlmc_grads_t,
                    mlmc_grads_tm1,
                )

                tracker_grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    tracker_grads,
                    tracker_grads_t,
                    tracker_grads_tm1,
                )
        else:
            (observation, env_state, env_params, key, policy_state, _), traj_batch  = (
            jax.lax.scan(
                step_fn,
                (observation, env_state, env_params, step_key, policy_state, jnp.array(0)),
                xs=None,
                length=jnp.minimum(max_t, batchsize_bound),
                )
            )

            (_, (av_reward, av_value)), tracker_grads = (
                reward_tracker_grad_fn(tracker_state.params, traj_batch.reward, traj_batch.value)
            )

            last_val = value(policy_state, observation)
            advantages, targets = _calculate_gae(traj_batch, av_reward, last_val)
            
            (loss_value, (value_loss, loss_actor, entropy)), grads = loss_grad_fn(
                policy_state.params, traj_batch, advantages, targets, av_value
            )


        policy_state = policy_state.apply_gradients(grads=grads)
        tracker_state = tracker_state.apply_gradients(grads=tracker_grads)
        grad_norm = optax.global_norm(grads)
        return (
            (loss_value, av_reward, av_value),
            grad_norm, 
            policy_state,
            tracker_state,
            (value_loss, loss_actor, entropy),
            observation,
            env_state,
            key
        )

    state, env_state = jit_reset(reset_key, env_params)

    for i_episode in range(1, n_training_episodes + 1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)

        (
            loss,
            grad_norm,
            policy_state,
            tracker_state,
            (value_loss, actor_loss, entropy),
            state,
            env_state,
            key,
        ) = ppo_update(
            policy_state,
            tracker_state,
            state,
            env_state,
            env_params,
            loss_key,
            step_key,
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)
        if "pogema" in env_id:
            metrics_deque.append(env_state.returned_episode_metrics)

        wandb.log(
            {
                "Loss": loss[0].mean().item(),
                "Grad_Norm": grad_norm.mean().item(),
                "Average_Reward_Tracker": loss[1].mean().item(),
                "Average_Value_Tracker": loss[2].mean().item(),
                "Value_Loss": value_loss.mean().item(),
                "Policy_Loss": actor_loss.mean().item(),
                "Entropy": entropy.mean().item(),
                "Step": i_episode,
                "Episode_Return": np.mean(scores_deque),
                "Episode_Length": np.mean(lengths_deque),
            }
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

    optimisers = {
        "sgd": optax.sgd(learning_rate=dict_config["learning_rate"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["learning_rate"]),
        "adam": optax.adam(learning_rate=dict_config["learning_rate"]),
        "adamW": optax.adamw(learning_rate=dict_config["learning_rate"]),
        "accelerated_sgd": accelerated_trace(
            learning_rate=lambda t: 0.1, **dict_config["momentum"]
        ),
        "accelerated_sgd_adagrad": optax.chain(
            optax.scale_by_rss(),
            accelerated_trace(learning_rate=lambda t: 0.1, **dict_config["momentum"]),
        ),
    }

    opt = optax.chain(
        optax.clip_by_global_norm(1000.0), optimisers[dict_config["optimiser"]]
    )
    run_actorcritic_experiment_ppo(
        optimiser=opt, av_tracker_optimiser=opt, **dict_config["experiment"], seed=dict_config["seed"]
    )


if __name__ == "__main__":
    main()
