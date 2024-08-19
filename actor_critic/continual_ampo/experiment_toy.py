import functools
import warnings
from collections import deque
from typing import Any,  Callable, NamedTuple, Optional

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import navix as nx
import numpy as np
import optax
import optax._src.base as base
import optax._src.utils as utils
import wandb
from distrax import Categorical
from flax import linen as nn
from flax.struct import dataclass
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")

class EnvState(NamedTuple):
    rng: jax.Array
    state_features: jax.Array
    features: jax.Array
    t: jax.Array


@dataclass
class RegressionMDP:
    n_states: int
    n_state_features: int
    state_dim: int
    p: float
    noise_std: float
    normalize_features: bool = False
    seed: int = None
    reward_func = jnp.matmul

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_env(self, rng):
        reset_rng, feat_rng, rng = jax.random.split(rng, 3)

        state_features, features = self.reset_features(feat_rng)
        initial_state = jax.random.randint(reset_rng, (1,), 0, self.n_states).astype(int)
        return initial_state, EnvState(rng, state_features, features, jnp.array(0))

    def reset(self, env_state: EnvState=None, seed=None) -> tuple[jax.Array, EnvState]:
        if seed is None and env_state is not None:
            reset_rng, _ = jax.random.split(env_state.rng)
        else:
            reset_rng, _ = jax.random.split(jax.random.key(seed))

        return self.reset_env(reset_rng)


    def reset_features(self, rng: jax.Array) -> jax.Array:
        w_rng, x_rng = jax.random.split(rng)
        w = jax.random.uniform(w_rng, (self.n_states, self.n_state_features, 1))
        w = w / (jnp.linalg.norm(w, axis=-1, ord=2, keepdims=True) + 1e-8)

        x = jax.random.normal(x_rng, (self.n_states, self.state_dim, self.n_state_features))
        if self.normalize_features:
            x = x / (jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True) + 1e-8)

        return x, w

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, prev_state: jax.Array, env_state: EnvState) -> tuple[jax.Array, EnvState]:

        step_rng, rng = jax.random.split(env_state.rng)
        next_state = jax.lax.cond(
            jax.random.bernoulli(step_rng, self.p),
            lambda x: jnp.mod(x+1, self.n_states),
            lambda x: x,
            prev_state
        )
        env_state = env_state._replace(rng=rng)
        env_state = env_state._replace(t=env_state.t+1)

        return next_state, env_state

def exponential_decay(
    lr: float,
    transition_steps: int
) -> base.Schedule:

    def schedule(count):
        return lr * jnp.sqrt(1 - (1 - count / transition_steps) ** count)

    return schedule


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


def accelerated_mgd(
    learning_rate: float,
    momentum: float = 1.0,
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

        beta_t = momentum / (1 + count/2.)
    
        trace_g = jax.tree_util.tree_map(
            lambda x, x_f: mul_by_alpha(beta_t, x)
            + mul_by_alpha(1.0 - beta_t, x_f),
            params,
            state.trace_f,
        )

        trace_f_update_fn = lambda g: learning_rate * g
        new_trace_f = jax.tree_util.tree_map(trace_f_update_fn, updates)

        updates = jax.tree_util.tree_map(
            lambda x, x_f: mul_by_alpha(beta_t, x)
            + mul_by_alpha(1.0 - beta_t, x_f),
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
        phi = lambda x: jnp.exp(x - 1),
        phi_inv = lambda x: jnp.log(x) + 1
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

def run_regression_experiment(
    mdp: RegressionMDP,
    loss_fn: Callable,
    optimiser: optax.GradientTransformation,
    num_iterations: int,
    mlmc_correction: bool,
    batchsize_limit: int,
    batchsize_bound: int,
    seed: int = None,
    **kwargs
):
    def collect_batch(carry, _):
        state, env_state = carry
        state, env_state = mdp.step(state, env_state)
        x = env_state.state_features[state[0]]
        w = env_state.features[state[0]]
        y = jnp.matmul(x, w) + mdp.noise_std * jax.random.normal(env_state.rng)
        return (state, env_state), (x, w, y)

    state, env_state = mdp.reset(seed=seed)

    # Creates a one linear layer instance.
    model = nn.Dense(features=1, use_bias=False, kernel_init=nn.initializers.glorot_uniform())
    # Initializes the parameters.
    rng = jax.random.PRNGKey(seed+1)
    params_rng, rng = jax.random.split(rng)
    params = model.init(params_rng, env_state.state_features[state[0]])

    # Creates a function that returns value and gradient of the loss.
    @jax.jit
    def batched_loss_fn(params, x, y):
        # Vectorises the squared error and computes mean over the loss values.
        y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
        per_example_diff = jnp.squeeze(jax.vmap(loss_fn)(y, y_pred), -1)
        return jnp.sum((jax.vmap(loss_fn)(y_pred, y)).squeeze(-1))/mdp.state_dim, per_example_diff

    loss_grad_fn = jax.value_and_grad(batched_loss_fn, has_aux=True)
    opt_state = optimiser.init(params)
    params_mean = params

    # Minimises the loss.
    def uncorrected(carry, _):

        state, env_state, opt_state, params, params_mean, iteration = carry
        (state, env_state), (x, w, y) = jax.lax.scan(collect_batch, (state, env_state), None, length=batchsize_bound)
        # Computes gradient of the loss.
        (_, mse), grads = loss_grad_fn(params, x, y)

        # Updates the optimiser state, creates an update to the params.
        updates, opt_state = optimiser.update(grads, opt_state, params)
        # Updates the parameters.
        params = optax.apply_updates(params, updates)
        y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params_mean, x)
        preds = (
            jnp.squeeze(jax.vmap(loss_fn)(y_pred, y), -1)
            -jnp.squeeze(jax.vmap(loss_fn)(jnp.matmul(x, w), y), -1)
        )

        params_mean = jax.tree_util.tree_map(
            lambda x, y: x+y,
            jax.tree_util.tree_map(lambda x: x*(iteration-1)/iteration, params_mean),
            jax.tree_util.tree_map(lambda x: x/iteration, params)
        )

        losses = mse
        iteration += 1

        return (state, env_state, opt_state, params, params_mean, iteration), (losses, preds)

    if mlmc_correction:
        iteration = 0
        num_observed_samples = 0
        preds, losses = [], []
        while True:
            iteration += 1

            initial_state = state
            (state, env_state), (x, w, y) = jax.lax.scan(collect_batch, (initial_state, env_state), None, length=batchsize_bound)

            num_observed_samples += (batchsize_bound * mdp.state_dim)
            #truncated geometric distribution
            rng, _ = jax.random.split(rng)
            N = 5
            p = 0.5
            #https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            # J = jax.random.geometric(rng, p)
            J = jnp.floor(jnp.log(1-jax.random.uniform(rng)*(1-(1-p)**N))/jnp.log(1-p)).astype(int)
            #mlmc batch_size
            (_, mse), grads = loss_grad_fn(params, x, y) #0

            if 2**J <= min(num_iterations+1, batchsize_limit) and J > 0: # >=2

                J_t, J_tm1 = jnp.ceil((2**J) * batchsize_bound), jnp.ceil((2**(J-1)) * batchsize_bound).astype(int)

                (state, env_state), (x, w, y) = jax.lax.scan(collect_batch, (initial_state, env_state), None, length=J_t)
                num_observed_samples += (J_t * mdp.state_dim)
                (_, mse), mlmc_grads_t = loss_grad_fn(params, x, y) #t
                _, mlmc_grads_tm1 = loss_grad_fn(params, x[:J_tm1], y[:J_tm1]) #tm1
                grads = jax.tree.map(lambda g_0, g_t, g_tm1: g_0 + (2**J) * (g_t - g_tm1), grads, mlmc_grads_t, mlmc_grads_tm1)

            # Updates the optimiser state, creates an update to the params.
            updates, opt_state = optimiser.update(grads, opt_state, params)
            # Updates the parameters.
            params = optax.apply_updates(params, updates)
            y_pred = jax.vmap(model.apply, in_axes=(None, 0))(params_mean, x)
            pred = (
                jnp.squeeze(jax.vmap(loss_fn)(y_pred, y), -1)
                -jnp.squeeze(jax.vmap(loss_fn)(jnp.matmul(x, w), y), -1)
            )

            preds.extend(pred.reshape(-1).tolist())

            params_mean = jax.tree_util.tree_map(
                lambda x, y: x+y,
                jax.tree_util.tree_map(lambda x: x*(iteration-1)/iteration, params_mean),
                jax.tree_util.tree_map(lambda x: x/iteration, params)
            )
            losses.extend(mse.reshape(-1).tolist())

            if num_observed_samples >= num_iterations * mdp.state_dim:
                break

    else:
        _, (losses, preds) = jax.lax.scan(uncorrected, (state, env_state, opt_state, params, params_mean, jnp.array(1)), None, length=num_iterations//batchsize_bound)
        losses, preds = losses.reshape(-1).tolist(), preds.reshape(-1).tolist()

    return losses[:num_iterations * mdp.state_dim], preds[:num_iterations * mdp.state_dim]

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

        "accelerated_sgd": accelerated_trace(
            learning_rate=lambda t: 0.1, **dict_config["momentum"]
        ),
        "accelerated_sgd_adagrad": optax.chain(
            optax.scale_by_rss(),
            accelerated_trace(learning_rate=lambda t: 0.1, **dict_config["momentum"]),
        ),
    }

    tracker_optimisers = {
        "mamd": accelerated_mgd(learning_rate=dict_config["alpha"], mlmc_correction=dict_config["experiment"]["mlmc_correction"]),
        "sgd": optax.sgd(learning_rate=dict_config["alpha"]),
        "adagrad": optax.adagrad(learning_rate=dict_config["alpha"]),
        # "adam": optax.adam(learning_rate=optax.exponential_decay(dict_config["alpha"], dict_config["experiment"]["n_training_episodes"], 0.99, 1)),
        "adam": optax.adam(learning_rate=dict_config["alpha"]),

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
