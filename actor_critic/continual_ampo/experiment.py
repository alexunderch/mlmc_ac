import functools
import warnings
from collections import deque
from typing import Any,  Literal, NamedTuple, Optional

import gymnax
import gymnax.wrappers
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import navix as nx
import numpy as np
import optax
import optax._src.base as base
import optax._src.utils as utils
from optax.contrib import (
    prodigy, 
    schedule_free_sgd, 
    schedule_free_adamw, 
    schedule_free
)
import wandb
from distrax import Categorical
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from omegaconf import DictConfig, OmegaConf


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
    policy_logits: jnp.ndarray
    log_prob: jnp.ndarray
    proj_lambdas: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    done: jnp.ndarray
    info: jnp.ndarray

class AcceleratedTraceState(NamedTuple):
    """State for the Adam algorithm."""

    count: jax.Array  # shape=(), dtype=jnp.int32.
    trace: base.Params


# def accelerated_trace(
#     learning_rate: float,
#     decay_theta: float,
#     decay_eta: float,
#     decay_beta: float,
#     decay_p: float,
#     accumulator_dtype: Optional[Any] = None,
# ) -> base.GradientTransformation:

#     accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

#     def init_fn(params):
#         return AcceleratedTraceState(
#             count=jnp.zeros((), dtype=jnp.int32), trace_f=params
#         )

#     def update_fn(updates, state, params):
#         count = state.count
#         mul_by_alpha = lambda alpha, x: jax.tree_util.tree_map(lambda y: alpha * y, x)

#         trace_g = jax.tree_util.tree_map(
#             lambda x, x_f: mul_by_alpha(decay_theta, x_f)
#             + mul_by_alpha(1.0 - decay_theta, x),
#             params,
#             state.trace_f,
#         )

#         trace_f_update_fn = lambda g, t: t - decay_p * learning_rate(count) * g
#         new_trace_f = jax.tree_util.tree_map(trace_f_update_fn, updates, trace_g)

#         updates = jax.tree_util.tree_map(
#             lambda x, x_f, next_x_f, x_g: (
#                 +mul_by_alpha(decay_eta, next_x_f)
#                 + mul_by_alpha(decay_p - decay_eta, x_f)
#                 + mul_by_alpha((1.0 - decay_p) * (1.0 - decay_beta) - 1.0, x)
#                 + mul_by_alpha((1.0 - decay_p) * decay_beta, x_g)
#             ),
#             params,
#             state.trace_f,
#             new_trace_f,
#             trace_g,
#         )
#         new_trace_f = utils.cast_tree(new_trace_f, accumulator_dtype)

#         return updates, AcceleratedTraceState(trace_f=new_trace_f, count=count + 1)

#     return base.GradientTransformation(init_fn, update_fn)


def accelerated_mgd(
    learning_rate: float,
    mlmc_correction: bool,
    momentum: float = 1.0,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:

    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)

    def init_fn(params):
        return AcceleratedTraceState(
            count=jnp.zeros((), dtype=jnp.int32), trace=params
        )

    def update_fn(updates, state, params):
        count = state.count
        mul_by_alpha = lambda alpha, x: jax.tree_util.tree_map(lambda y: alpha * y, x)

        beta_t = momentum / (1 + (count - 500.01 * int(not mlmc_correction)) / 2.)
        gamma_t = 1 + (count - 500.01 * int(not mlmc_correction)) / 2.


        updates = jax.tree.map(lambda g: -learning_rate * gamma_t * g, updates)

        new_trace = jax.tree.map(lambda g, t: t + g, params, updates)
        
        updates = jax.tree_util.tree_map(
            lambda x, x_f: mul_by_alpha(beta_t, x)
            + mul_by_alpha(- beta_t, x_f),
            state.trace,
            params,
        )
        new_trace = utils.cast_tree(new_trace, accumulator_dtype)
 
        return updates, AcceleratedTraceState(trace=new_trace, count=count + 1)

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
    return nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(primals, tangents):
    (x, ) = primals
    (x_dot, ) = tangents
    primal_out = _projection_unit_simplex(x)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp)
    tangent_out = jnp.dot(jnp.diag(supp) - jnp.outer(supp, supp) / card, x_dot)
    return primal_out, tangent_out


def _projection_simplex(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
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

    proj = _projection_unit_simplex(jnp.exp(x) / value)
    return value * proj

def projection_simplex(
        x: jnp.ndarray, num_actions: int, 
        phi, phi_inv
    ):
    a = phi_inv(1)
    b = phi_inv(1 / num_actions)
    nu1 = a - jnp.max(x)
    nu2 = b - jnp.max(x)
    for i in range(10):
        nu = (nu1 + nu2) / 2
        cond = jnp.sum(nn.relu(phi(x + nu))) > 1
        nu1 = jnp.where(cond, nu, nu1)
        nu2 = jnp.where(1.0 - cond, nu, nu2)
    projected_x = nn.relu(phi(x + nu1))
    projected_x = projected_x / jnp.sum(projected_x)
    return projected_x, nu1



def run_actorcritic_experiment_mdpo(
    env_id: str,
    num_envs: int,
    num_minibatches: int,
    optimiser: optax.GradientTransformation,
    critic_optimiser: optax.GradientTransformation,
    av_tracker_optimiser: optax.GradientTransformation,
    n_training_episodes: int,
    gae_lambda: float,
    vf_coeff: float,
    ent_coeff: float,
    av_vf_coeff: float,
    clip_eps: float,
    seed: int,
    batchsize_bound: int,
    batchsize_limit: int,
    mlmc_correction: bool,
    projection: Optional[str] = "simplex",
    total_samples: Optional[int] = None,
    normalise_advantages: Optional[bool] = True,
    mixing_parameters_param: Optional[bool] = False
):
    
    if projection == "simplex":
        phi = lambda x: jnp.exp(x - 1)
        phi_inv = lambda x: jnp.log(x) + 1
    else:
        phi = lambda x: x
        phi_inv = lambda x: x

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

    @jax.jit
    def _act_smplx(policy_state, observations, key):
        action_logits = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))(
            {"params": policy_state.params}, observations
        )
        lr_actor = jnp.where(
            mixing_parameters_param,
            1e-3 * (policy_state.step/2. + 1),
            1.0
        )
        action_logits *= lr_actor

        probs, lambda_ = jax.vmap(projection_simplex, in_axes=(0, None, None, None))(
            action_logits, 
            policy_network.num_actions,
            phi, phi_inv
        )
        pi = Categorical(probs=probs)
        action, log_prob = pi.sample_and_log_prob(seed=key)

        return action, action_logits, log_prob, lambda_
    act = _act_smplx


    @jax.jit
    def value(critic_state, observations):
        # observations = jnp.reshape(observations, (num_envs, -1))
        value = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(
            {"params": critic_state.params}, observations
        )
        return value

    if "navix" in env_id:
        metrics_key = "Nil"
        _, env_name = env_id.split(":")
        env, env_params = NavixGymnaxWrapper(env_name), None
        env = gymnax.wrappers.LogWrapper(env)

        jit_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        jit_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, critic_state, ep_len = carry
            env_step_key, act_key, key = jax.random.split(step_key, 3)
            state = jnp.reshape(state, (num_envs, -1))
            env_step_keys = jax.random.split(env_step_key, num_envs)
            action, policy_logits, log_prob, plambda = act(policy_state, state, act_key)
            value_ = value(critic_state, state)
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
                    critic_state,
                    ep_len + (1 - done.any().astype(jnp.int32)),
                ),
                Transition(state, action, policy_logits, log_prob, plambda, reward, value_, done, info),
            )

    else:
        metrics_key = "Nil"
        env, env_params = gymnax.make(env_id)
        env = gymnax.wrappers.LogWrapper(env)

        jit_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
        jit_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

        def step_fn(carry, _):
            state, env_state, env_params, step_key, policy_state, critic_state, ep_len = carry
            env_step_key, act_key, key = jax.random.split(step_key, 3)
            env_step_keys = jax.random.split(env_step_key, num_envs)
            action, policy_logits, log_prob, plambda = act(policy_state, state, act_key)
            value_ = value(critic_state, state)
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
                    critic_state,
                    ep_len + (1 - done.any().astype(jnp.int32)),
                ),
                Transition(state, action, policy_logits, log_prob, plambda, reward, value_, done, info),
            )

    class Policy(nn.Module):
        num_actions: jnp.ndarray

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # policy_logits = nn.Sequential([nn.Dense(features=self.num_actions)])(x)
            policy_logits = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            policy_logits = nn.tanh(policy_logits)

            policy_logits = nn.Dense(
                self.num_actions,
                kernel_init=nn.initializers.orthogonal(0.01),
                bias_init=nn.initializers.constant(0.0),
            )(policy_logits)

            return policy_logits
        
    class Critic(nn.Module):

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # values = nn.Sequential([nn.Dense(1)])(x)
            values = nn.Dense(
                64,
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.constant(0.0),
            )(x)
            values = nn.tanh(values)

            values = nn.Dense(
                1,
                kernel_init=nn.initializers.orthogonal(1.0),
                bias_init=nn.initializers.constant(0.0),
            )(values)

            return jnp.squeeze(values, axis=-1)

    class Tracker(nn.Module):
        output_dim: int

        @nn.compact
        def __call__(self):
            return (
                self.param(
                    "tracked_reward",
                    lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
                    (self.output_dim,),
                ),
                self.param(
                    "tracked_value",
                    lambda rng, shape: jnp.zeros(shape, dtype=jnp.float32),
                    (self.output_dim,),
                ),
            )

    av_tracker = Tracker(1)
    policy_network = Policy(env.num_actions)
    critic_network = Critic()

    init_policy_key, init_critic_key, init_reward_key, reset_key, key = jax.random.split(
        jax.random.key(seed), 5
    )
    initial_obs, env_state = jit_reset(jax.random.split(reset_key, 1), env_params)
    initial_obs = jnp.reshape(initial_obs, (-1,))

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy_network.apply),
        params=policy_network.init(init_policy_key, initial_obs)["params"],
        tx=optimiser,
    )

    critic_state = TrainState.create(
        apply_fn=jax.jit(critic_network.apply),
        params=critic_network.init(init_critic_key, initial_obs)["params"],
        tx=critic_optimiser,
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

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def _actor_loss_fn(params, traj_batch, gae):
        lr_actor = jnp.where(
            mixing_parameters_param,
            1e-3 * ((policy_state.step - 500.01 * int(not mlmc_correction))/2. + 1),
            1.0
        )
        previous_lr_actor = jnp.where(
            mixing_parameters_param,
             max(1e-3 * (1.0 + (policy_state.step-1. - 500.01 * int(not mlmc_correction))/2.), 1.0),
            1.0
        )
        
        
        # GRAIDENT OPTIMISATION PART
        # RERUN NETWORK
        action_logits = jax.vmap(policy_state.apply_fn, in_axes=(None, 0))(
            {"params": params}, traj_batch.observation
        )
        probs, _ = jax.vmap(projection_simplex, in_axes=(0, None, None, None))(
            action_logits, 
            policy_network.num_actions,
            phi, phi_inv
        )

        pi = Categorical(probs=probs * lr_actor)
        entropy = pi.entropy()
        log_prob = pi.log_prob(traj_batch.action.squeeze(-1))

        # CALCULATE ACTOR LOSS
        logratio = log_prob - traj_batch.log_prob.squeeze(-1)
        action_logits = action_logits[
            jnp.arange(len(action_logits)), traj_batch.action
        ]

        # CALCULATE ACTOR LOSS
        gae = jnp.where(
            normalise_advantages,
            (gae - gae.mean()) / (gae.std() + 1e-8),
            gae,
        )


        ############################################################        
        preproj = traj_batch.policy_logits * (previous_lr_actor / lr_actor)
        preproj = (
            preproj[jnp.arange(len(preproj)), traj_batch.action] 
             + jnp.squeeze(traj_batch.proj_lambdas)
        )

        objective = gae + jax.lax.stop_gradient(jnp.maximum(preproj, phi_inv(1e-6)) 
        ) 

        loss_actor = optax.l2_loss(action_logits, objective).mean() - ent_coeff * entropy.mean()
        ############################################################        


        return loss_actor, (loss_actor, entropy.mean(), logratio.mean())
    
    def _critic_loss_fn(params, traj_batch, targets, av_value):
        # GRAIDENT OPTIMISATION PART
        # RERUN NETWORK
        value = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(
            {"params": params}, traj_batch.observation
        )

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value.squeeze(-1) + (value - traj_batch.value).clip(
            -clip_eps, clip_eps
        )

        value = value - av_value * av_vf_coeff

        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

        total_loss = vf_coeff * value_loss 
        return total_loss, (value_loss, )

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
        critic_state,
        tracker_state,
        observation,
        env_state,
        env_params,
        rng,
        env_rng,
        sample_counter,
        update_counter,
    ):

        actor_loss_grad_fn = jax.value_and_grad(
            _actor_loss_fn,
            has_aux=True,
        )

        critic_loss_grad_fn = jax.value_and_grad(
            _critic_loss_fn,
            has_aux=True,
        )

        reward_tracker_grad_fn = jax.value_and_grad(
            _av_reward_tracker_loss_fn, has_aux=True,
        )

        def mix_params(train_state: TrainState):
            x_f = train_state.params
            if mixing_parameters_param:
                x, t = train_state.opt_state[1].trace, train_state.step
                beta = 1./(1. + (t - (500.01 * int(not mlmc_correction)))/2.)
                return optax.tree_utils.tree_add(
                    optax.tree_utils.tree_scalar_mul(beta, x),
                    optax.tree_utils.tree_scalar_mul(1. - beta, x_f),                    
                )
            return x_f


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
            ) + 1

            J_t, J_tm1 = int(jnp.ceil((2 ** J) * batchsize_bound)), int(
                jnp.ceil((2 ** (J - 1)) * batchsize_bound)
            )

            batchsize = J_t if 2**J <= batchsize_limit else batchsize_bound

            (observation, env_state, env_params, key, policy_state, critic_state, _), traj_batch = (
                jax.lax.scan(
                    step_fn,
                    (
                        observation,
                        env_state,
                        env_params,
                        env_rng,
                        policy_state,
                        critic_state,
                        jnp.array(0),
                    ),
                    xs=None,
                    length=batchsize,
                    unroll=False,
                )
            )

            (_, (av_reward, av_value)), tracker_grads = reward_tracker_grad_fn(
                tracker_state.params,
                traj_batch.reward,
                traj_batch.value,
            )
            
            last_obs = observation if (2**J <= batchsize_limit) else traj_batch.observation[batchsize_bound-1]
            last_val = value(critic_state, last_obs.reshape((num_envs, -1)))
            advantages, targets = _calculate_gae(traj_batch, last_val, av_reward)

            (actor_loss_value, (policy_loss, entropy, kl)), actor_grads = actor_loss_grad_fn(
                mix_params(policy_state),
                jax.tree.map(lambda x: x[:batchsize_bound].reshape((num_envs * batchsize_bound, -1)), traj_batch),
                advantages[:batchsize_bound].reshape(-1),
            ) #0


            (critic_loss_value, (value_loss, )), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                jax.tree.map(lambda x: x[:batchsize_bound].reshape((num_envs * batchsize_bound, -1)), traj_batch),
                targets[:batchsize_bound].reshape(-1),
                av_value[:batchsize_bound].reshape(-1),
            ) #0

            # mlmc batch_size
            if 2**J <= batchsize_limit:
                sample_counter += J_t * num_envs

                _, tracker_grads_t = reward_tracker_grad_fn(
                    tracker_state.params, traj_batch.reward, traj_batch.value
                )

                _, actor_grads_t = actor_loss_grad_fn(
                    mix_params(policy_state),
                    jax.tree.map(lambda x: x.reshape((num_envs * J_t, -1)), traj_batch),
                    advantages.reshape((J_t * num_envs, -1)),
                ) #t


                _, critic_grads_t = critic_loss_grad_fn(
                    critic_state.params,
                    jax.tree.map(lambda x: x.reshape((num_envs * J_t, -1)), traj_batch),
                    targets.reshape((J_t * num_envs, -1)),
                    av_value.reshape(-1),
                ) #t

                _, tracker_grads_tm1 = reward_tracker_grad_fn(
                    tracker_state.params,
                    traj_batch.reward[:J_tm1],
                    traj_batch.value[:J_tm1],
                )

                _, actor_grads_tm1 = actor_loss_grad_fn(
                    mix_params(policy_state),
                    jax.tree.map(lambda x: x[:J_tm1].reshape((num_envs * J_tm1, -1)), traj_batch),
                    advantages[:J_tm1].reshape(-1),
                ) #0


                _, critic_grads_tm1 = critic_loss_grad_fn(
                    critic_state.params,
                    jax.tree.map(lambda x: x[:J_tm1].reshape((num_envs * J_tm1, -1)), traj_batch),
                    targets[:J_tm1].reshape(-1),
                    av_value[:J_tm1].reshape(-1),
                ) #tm1

                actor_grads = jax.tree.map(
                    lambda g, x, y: g +(2**J) * (x - y),
                    actor_grads,
                    actor_grads_t,
                    actor_grads_tm1,
                )

                critic_grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    critic_grads,
                    critic_grads_t,
                    critic_grads_tm1,
                )

                tracker_grads = jax.tree.map(
                    lambda g, x, y: g + (2**J) * (x - y),
                    tracker_grads,
                    tracker_grads_t,
                    tracker_grads_tm1,
                )

            else:
                sample_counter += batchsize_bound * num_envs
        else:
            sample_counter += batchsize_bound * num_envs

            (observation, env_state, env_params, key, policy_state, critic_state, _), traj_batch = (
                jax.lax.scan(
                    step_fn,
                    (
                        observation,
                        env_state,
                        env_params,
                        step_key,
                        policy_state,
                        critic_state,
                        jnp.array(0),
                    ),
                    xs=None,
                    length=batchsize_bound,
                )
            )

            (_, (av_reward, av_value)), tracker_grads = reward_tracker_grad_fn(
                tracker_state.params, traj_batch.reward, traj_batch.value
            )
            last_val = value(critic_state, observation.reshape((num_envs, -1)))
            advantages, targets = _calculate_gae(traj_batch, last_val, av_reward)

            traj_batch = jax.tree.map(
                lambda x: jnp.reshape(x, (batchsize_bound * num_envs, -1)), traj_batch
            )
            
            (advantages, targets) = jax.tree.map(
                lambda x: jnp.reshape(x, (batchsize_bound * num_envs, -1)), (advantages, targets)
            )

            (actor_loss_value, (policy_loss, entropy, kl)), actor_grads = actor_loss_grad_fn(
                mix_params(policy_state),
                traj_batch,
                advantages,
            )

            (critic_loss_value, (value_loss, )), critic_grads = critic_loss_grad_fn(
                critic_state.params,
                traj_batch,
                targets,
                av_value.reshape(-1),
            )

        # jax.debug.print("{x}", x=grads)
        policy_state = policy_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        tracker_state = tracker_state.apply_gradients(grads=tracker_grads)
        
        actor_grad_norm = optax.global_norm(actor_grads)
        critc_grad_norm = optax.global_norm(critic_grads)
        loss_value = actor_loss_value + critic_loss_value
        return (
            (loss_value, av_reward, av_value),
            (actor_grad_norm, critc_grad_norm),
            policy_state,
            critic_state,
            tracker_state,
            (value_loss, policy_loss, entropy, kl),
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
            critic_state,
            tracker_state,
            (value_loss, actor_loss, entropy, kl),
            state,
            env_state,
            key,
            sample_counter,
        ) = _update(
            policy_state,
            critic_state,
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
                "Actor_Grad_Norm": grad_norm[0].mean().item(),
                "Critic_Grad_Norm": grad_norm[1].mean().item(),
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
                "env_samples": sample_counter
            },
        )

def exponential_decay(
    lr: float,
    transition_steps: int
) -> base.Schedule:


  def schedule(count):
    return lr * jnp.sqrt(1 - (1 - count / transition_steps) ** count)

  return schedule

@hydra.main(config_path=".", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:

    dict_config = wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project
        )
    
    adam_schedule = exponential_decay(
        dict_config["learning_rate"], 
        dict_config["experiment"]["n_training_episodes"]
    )
    adam_opt = optax.chain(
    optax.scale_by_adam(b1=0, b2=0.9999),  # Use the updates from adam.
    optax.scale_by_schedule(adam_schedule),  # Use the learning rate from the scheduler.
    # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    optax.scale(-1.0)
)
    optimisers = {
        "mamd": accelerated_mgd(learning_rate=dict_config["learning_rate"], mlmc_correction=dict_config["experiment"]["mlmc_correction"]),
        "sgd": optax.sgd(learning_rate=dict_config["learning_rate"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["learning_rate"]),
        "adam": adam_opt, #optax.adam(learning_rate=dict_config["learning_rate"]),
        "adamw": optax.adamw(learning_rate=dict_config["learning_rate"]),
        "prodigy": prodigy(learning_rate=dict_config["learning_rate"]),
        "sf_mamd": schedule_free(accelerated_mgd(learning_rate=0.01, mlmc_correction=dict_config["experiment"]["mlmc_correction"]), dict_config["learning_rate"]/100),
        "sf_sgd": schedule_free_sgd(learning_rate=dict_config["learning_rate"]),
        "sf_adamW": schedule_free_adamw(learning_rate=dict_config["learning_rate"]),

        # "accelerated_sgd": accelerated_trace(
        #     learning_rate=lambda t: 0.1, **dict_config["momentum"]
        # ),
        # "accelerated_sgd_adagrad": optax.chain(
        #     optax.scale_by_rss(),
        #     accelerated_trace(learning_rate=lambda t: 0.1, **dict_config["momentum"]),
        # ),
    }

    tracker_optimisers = {
        "mamd": accelerated_mgd(learning_rate=dict_config["alpha"], mlmc_correction=dict_config["experiment"]["mlmc_correction"]),
        "sgd": optax.sgd(learning_rate=dict_config["alpha"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["alpha"]),
        # "adam": optax.adam(learning_rate=optax.exponential_decay(dict_config["alpha"], dict_config["experiment"]["n_training_episodes"], 0.99, 1)),
        "adam": optax.adam(learning_rate=dict_config["alpha"]),
        "adamw": optax.adamw(learning_rate=dict_config["alpha"]),
        "prodigy": prodigy(learning_rate=dict_config["alpha"]),
        "sf_mamd": schedule_free(accelerated_mgd(learning_rate=1.0, mlmc_correction=dict_config["experiment"]["mlmc_correction"]), dict_config["learning_rate"]),
        "sf_sgd": schedule_free_sgd(learning_rate=dict_config["alpha"]),
        "sf_adamw": schedule_free_adamw(learning_rate=dict_config["alpha"]),
        # "accelerated_sgd": accelerated_trace(
        #     learning_rate=lambda t: dict_config["alpha"], **dict_config["momentum"]
        # ),
        # "accelerated_sgd_adagrad": optax.chain(
        #     optax.scale_by_rss(),
        #     accelerated_trace(
        #         learning_rate=lambda t: dict_config["alpha"], **dict_config["momentum"]
        #     ),
        # ),
    }


    opt = optax.chain(
        optax.clip_by_global_norm(0.5), optimisers[dict_config["optimiser"]]
    )


    critic_opt = optax.chain(
        optax.clip_by_global_norm(0.5), tracker_optimisers[dict_config["critic_optimiser"]]
    )

    opt_tracker = optax.chain(
        optax.clip_by_global_norm(0.5), tracker_optimisers[dict_config["critic_optimiser"]]
    )
    run_actorcritic_experiment_mdpo(
        optimiser=opt,
        critic_optimiser=critic_opt,
        av_tracker_optimiser=opt_tracker,
        **dict_config["experiment"],
        seed=dict_config["seed"],
        mixing_parameters_param="mamd" in dict_config["optimiser"]
    )


if __name__ == "__main__":
    main()
