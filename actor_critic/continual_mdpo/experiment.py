import functools
import warnings
from collections import deque
from typing import Any, Callable, Literal, NamedTuple, Optional

import gymnax
from gymnax.environments import spaces
import gymnax.wrappers
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import navix as nx
import numpy as np
import optax
import optax._src.base as base
import optax._src.utils as utils
from distrax import Categorical
from distrax._src.utils import math
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState
# from pogema.envs import _make_pogema
# from pogema.grid_config import GridConfig
# from pogema.integrations.sample_factory import AutoResetWrapper

warnings.filterwarnings("ignore")


DEBUG_POGEMA = False


class NavixGymnaxWrapper:
    def __init__(self, env_name):
        self._env = nx.make(env_name)

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )

    @property
    def num_actions(self):
        return self._env.action_space.maximum.item() + 1


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray


def isr_decay(initial_value: float) -> base.Schedule:
    """
    Args:
      initial_value: value to decay from.

    Returns:
      schedule
        A function that maps step counts to values.
    """
    return lambda count: initial_value / (count + 1.0)


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


@jax.custom_jvp
def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
    """Projection onto the unit simplex."""
    s = 1.0
    n_features = x.shape[0]
    u = jnp.sort(x)[::-1]
    cumsum_u = jnp.cumsum(u)
    ind = jnp.arange(n_features) + 1
    cond = s / ind + (u - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond)
    return jax.nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = _projection_unit_simplex(x)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp)
    tangent_out = jnp.dot(jnp.diag(supp) - jnp.outer(supp, supp) / card, x_dot)
    return primal_out, tangent_out


def projection_simplex(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
    r"""Projection onto a simplex:

    .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{value}

    By default, the projection is onto the probability simplex.

    Args:
    x: vector to project, an array of shape (n,).
    value: value p should sum to (default: 1.0).
    Returns:
    projected vector, an array with the same shape as ``x``.
    """
    if value is None:
        value = 1.0
    return value * _projection_unit_simplex(x / value)


def run_actorcritic_experiment_mdpo(
    env_id: str,
    num_envs: int,
    optimiser: optax.GradientTransformation,
    av_tracker_optimiser: optax.GradientTransformation,
    projection: Literal["softmax", "simplex"],
    n_training_episodes: int,
    max_t: int,
    gae_lambda: float,
    vf_coeff: float,
    ent_coeff: float,
    av_vf_coeff: float,
    clip_eps: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool,
    total_samples: Optional[int] = None,
    env_kwargs: Optional[dict] = None,
):
    scores_deque = deque(maxlen=100)
    lengths_deque = deque(maxlen=100)
    metrics_deque = deque(maxlen=100)
    batchsize_bound *= num_envs


    total_samples: int = (
        int(total_samples)
        if total_samples is not None
        else batchsize_bound * n_training_episodes
    )

    total_samples /= num_envs
    n_training_episodes /= num_envs

    sample_counter: int = 0
    stopping_criterium = lambda k: k > total_samples
    temperature_schedule = isr_decay(1.0)

    @jax.jit
    def _act_cat(policy_state, observations, key):
        action_logits, value = policy_state.apply_fn(
            {"params": policy_state.params}, observations
        )
        pi = Categorical(logits=action_logits)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value

    @jax.jit
    def _act_smplx(policy_state, observations, key):
        action_logits, value = policy_state.apply_fn(
            {"params": policy_state.params}, observations
        )
        pi = Categorical(probs=projection_simplex(action_logits))
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        return action, log_prob, value

    if projection == "simplex":
        act = _act_smplx
    elif projection == "softmax":
        act = _act_cat

    @jax.jit
    def value(policy_state, observations):
        observations = jnp.reshape(observations, (-1,))
        _, value = policy_state.apply_fn({"params": policy_state.params}, observations)
        return value

    if "navix" in env_id:
        metrics_key = "Nil"
        _, env_name = env_id.split(":")
        env, env_params = NavixGymnaxWrapper(env_name), None
        env = gymnax.wrappers.LogWrapper(env)

        jit_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        jit_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, ep_len = carry
            env_step_key, act_key, key = jax.random.split(step_key, 3)
            state = jnp.reshape(state, (num_envs, -1))
            env_step_keys = jax.random.split(env_step_key, config["NUM_ENVS"])
            action, log_prob, value = act(policy_state, state, act_key)
            next_state, new_env_state, reward, done, info = jit_step(
                env_step_keys, env_state, action, env_params
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

    elif "pogema" not in env_id:
        metrics_key = "Nil"
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
            # policy_logits = nn.Sequential([nn.Dense(features=self.num_actions)])(x)
            # values = nn.Sequential([nn.Dense(1)])(x)
            policy_logits = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            policy_logits = nn.tanh(policy_logits)
            policy_logits = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(policy_logits)
            policy_logits = nn.tanh(policy_logits)
            policy_logits = nn.Dense(
                self.num_actions,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            )(policy_logits)

            values = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            values = nn.tanh(values)
            values = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(values)
            values = nn.tanh(values)
            values = nn.Dense(
                1,
                kernel_init=nn.initializers.orthogonal(1.0),
                bias_init=nn.initializers.constant(0.0),
            )(values)

            return policy_logits, jnp.squeeze(values, axis=-1)

    class Tracker(nn.Module):
        output_dim: int

        @nn.compact
        def __call__(self):
            return (
                self.param(
                    "tracked_reward",
                    lambda rng, shape: jnp.zeros(shape),
                    (self.output_dim,),
                ),
                self.param(
                    "tracked_value",
                    lambda rng, shape: jnp.zeros(shape),
                    (self.output_dim,),
                ),
            )

    av_tracker = Tracker(1)
    network = ActorCritic(
        env.num_actions if "pogema" not in env_id else env.action_space.n
    )

    init_network_key, init_reward_key, reset_key, key = jax.random.split(
        jax.random.key(seed), 4
    )
    initial_obs, env_state = jit_reset(reset_key, env_params)
    initial_obs = jnp.reshape(initial_obs, (-1,))

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
            delta = (
                reward - av_reward.astype(jnp.float32) + next_value * (1 - done) - value
            )
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

    def _loss_fn(params, temperature, traj_batch, gae, targets, av_value):
        # GRAIDENT OPTIMISATION PART
        # RERUN NETWORK
        action_logits, value = policy_state.apply_fn(
            {"params": params}, traj_batch.observation
        )

        if projection == "simplex":
            # jax.debug.print(
            # "{y}\t{x}", y=action_logits[0], x=projection_simplex(action_logits[0])
            # )
            pi = Categorical(probs=jax.vmap(projection_simplex)(action_logits))
            # jax.debug.print("{x}", x=pi.probs[0])
            # pi = Categorical(logits=action_logits)

        elif projection == "softmax":
            pi = Categorical(logits=action_logits)

        log_prob = pi.log_prob(traj_batch.action)
        entropy = jnp.nanmean(pi.entropy())

        # CALCULATE VALUE LOSS
        value = value - av_value * av_vf_coeff
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -clip_eps, clip_eps
        )
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        logratio = log_prob - traj_batch.log_prob
        ratio = jnp.exp(logratio)
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
        loss_actor = loss_actor + temperature * logratio
        loss_actor = loss_actor.mean()

        total_loss = loss_actor + vf_coeff * value_loss - ent_coeff * entropy
        return total_loss, (value_loss, loss_actor, entropy, logratio.mean())

    @jax.jit
    def _av_reward_tracker_loss_fn(params, returns, values):
        apply_fn = tracker_state.apply_fn
        av_reward, av_value = apply_fn({"params": params})
        av_reward_update = optax.l2_loss(av_reward.squeeze(), returns.mean())
        av_value_update = optax.l2_loss(av_value.squeeze(), values.mean())
        total_loss = av_reward_update + av_value_update
        return (
            total_loss,
            (av_reward, jnp.tile(av_value, values.shape)),
        )

    def _update(
        policy_state,
        tracker_state,
        observation,
        env_state,
        env_params,
        rng,
        env_rng,
        sample_counter,
        update_counter,
    ):

        loss_grad_fn = jax.value_and_grad(
            _loss_fn,
            has_aux=True,
        )
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

            J_t, J_tm1 = int(jnp.ceil((2**J) * batchsize_bound * num_envs)), int(
                jnp.ceil((2 ** (J - 1)) * batchsize_bound * num_envs)
            )
            (observation, env_state, env_params, key, policy_state, _), traj_batch = (
                jax.lax.scan(
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
            )

            traj_batch = jax.tree.map(lambda x: jnp.reshape(x, (num_envs*J_t, -1)), traj_batch)

            (_, (av_reward, av_value)), tracker_grads = reward_tracker_grad_fn(
                tracker_state.params,
                traj_batch.reward[:batchsize_bound],
                traj_batch.value[:batchsize_bound],
            )

            last_val = value(policy_state, observation.reshape((num_envs*batchsize_bound, -1)))
            advantages, targets = _calculate_gae(traj_batch, av_reward, last_val)

            (loss_value, (value_loss, loss_actor, entropy, kl)), grads = loss_grad_fn(
                policy_state.params,
                temperature_schedule(update_counter),
                jax.tree.map(lambda x: x[:batchsize_bound], traj_batch),
                advantages[:batchsize_bound],
                targets[:batchsize_bound],
                av_value[:batchsize_bound],
            )  # 0

            # mlmc batch_size
            if 2**J <= batchsize_limit and J > 0:
                sample_counter += J_t 

                (_, (av_reward, av_value)), tracker_grads_t = reward_tracker_grad_fn(
                    tracker_state.params, traj_batch.reward, traj_batch.value
                )

                (loss_value, (value_loss, loss_actor, entropy, kl)), mlmc_grads_t = (
                    loss_grad_fn(
                        policy_state.params,
                        temperature_schedule(update_counter),
                        traj_batch,
                        advantages,
                        targets,
                        av_value,
                    )
                )  # t

                _, tracker_grads_tm1 = reward_tracker_grad_fn(
                    tracker_state.params,
                    traj_batch.reward[:J_tm1],
                    traj_batch.value[:J_tm1],
                )

                _, mlmc_grads_tm1 = loss_grad_fn(
                    policy_state.params,
                    temperature_schedule(update_counter),
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
                sample_counter += batchsize_bound 
        else:
            sample_counter += batchsize_bound 

            (observation, env_state, env_params, key, policy_state, _), traj_batch = (
                jax.lax.scan(
                    step_fn,
                    (
                        observation,
                        env_state,
                        env_params,
                        step_key,
                        policy_state,
                        jnp.array(0),
                    ),
                    xs=None,
                    length=jnp.minimum(max_t, batchsize_bound),
                )
            )

            traj_batch = jax.tree.map(lambda x: jnp.reshape(x, (num_envs * J_t, -1)), traj_batch)

            (_, (av_reward, av_value)), tracker_grads = reward_tracker_grad_fn(
                tracker_state.params, traj_batch.reward, traj_batch.value
            )

            last_val = value(policy_state, observation.reshape((num_envs*batchsize_bound, -1)))
            advantages, targets = _calculate_gae(traj_batch, av_reward, last_val)

            (loss_value, (value_loss, loss_actor, entropy, kl)), grads = loss_grad_fn(
                policy_state.params,
                temperature_schedule(update_counter),
                traj_batch,
                advantages,
                targets,
                av_value,
            )
        
        # jax.debug.print("{x}", x=grads)
        policy_state = policy_state.apply_gradients(grads=grads)
        tracker_state = tracker_state.apply_gradients(grads=tracker_grads)
        grad_norm = optax.global_norm(grads)
        return (
            (loss_value, av_reward, av_value),
            grad_norm,
            policy_state,
            tracker_state,
            (value_loss, loss_actor, entropy, kl),
            observation,
            env_state,
            key,
            sample_counter,
        )

    state, env_state = jit_reset(jax.random.split(reset_key, num_envs), env_params)
    update_counter = 0
    for i_episode in range(1, n_training_episodes + 1):
        loss_key, reset_key, step_key = jax.random.split(key, 3)

        if stopping_criterium(sample_counter):
            break

        (
            loss,
            grad_norm,
            policy_state,
            tracker_state,
            (value_loss, actor_loss, entropy, kl),
            state,
            env_state,
            key,
            sample_counter,
        ) = _update(
            policy_state,
            tracker_state,
            state,
            env_state,
            env_params,
            loss_key,
            step_key,
            sample_counter,
            update_counter,
        )

        scores_deque.append(env_state.returned_episode_returns)
        lengths_deque.append(env_state.returned_episode_lengths)
        if "pogema" in env_id:
            metrics_deque.append(env_state.returned_episode_metrics)

        update_counter += 1

        wandb.log(
            {
                "Loss": loss[0].mean().item(),
                "Grad_Norm": grad_norm.mean().item(),
                "Average_Reward_Tracker": loss[1].mean().item(),
                "Average_Value_Tracker": loss[2].mean().item(),
                "KL": kl.mean().item(),
                "Value_Loss": value_loss.mean().item(),
                "Policy_Loss": actor_loss.mean().item(),
                "Entropy": entropy.mean().item(),
                "Step": i_episode,
                "Episode_Return": np.mean(scores_deque),
                "Episode_Length": np.mean(lengths_deque),
                f"Episode_{metrics_key}": np.mean(metrics_deque),
                "env_steps": sample_counter,
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

    wandb.define_metric("env_steps")

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

    tracker_optimisers = {
        "sgd": optax.sgd(learning_rate=dict_config["alpha"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["alpha"]),
        "adam": optax.adam(learning_rate=dict_config["alpha"]),
        "adamW": optax.adamw(learning_rate=dict_config["alpha"]),
        "accelerated_sgd": accelerated_trace(
            learning_rate=lambda t: dict_config["alpha"], **dict_config["momentum"]
        ),
        "accelerated_sgd_adagrad": optax.chain(
            optax.scale_by_rss(),
            accelerated_trace(
                learning_rate=lambda t: dict_config["alpha"], **dict_config["momentum"]
            ),
        ),
    }

    opt = optax.chain(
        optax.clip_by_global_norm(1000.0), optimisers[dict_config["optimiser"]]
    )

    opt_tracker = optax.chain(
        optax.clip_by_global_norm(1000.0), tracker_optimisers[dict_config["optimiser"]]
    )
    run_actorcritic_experiment_mdpo(
        optimiser=opt,
        av_tracker_optimiser=opt_tracker,
        **dict_config["experiment"],
        seed=dict_config["seed"],
    )


if __name__ == "__main__":
    main()
