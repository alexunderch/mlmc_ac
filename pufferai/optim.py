import torch
from torch.optim.optimizer import Optimizer, required
from torch.optim import SGD 
import copy
from typing import Literal, Any

from torch.func import functional_call, vmap, grad

class MLMCWrapper:
    def __init__(
        self, 
        model: torch.nn.Module,
        optimizer: Optimizer, 
        batchsize_bound: int, 
        batchsize_limit: int,
        gen_dist: Literal["geometric", "trunc_geometric"],
        geom_bins: int = 5,

    ) -> None:
        """
        """
        self.model = model
        self.sample_grads = None

        self.optim = optimizer
        self._B = batchsize_bound
        self._M = batchsize_limit
        self.J = 1

        N = geom_bins
        p = 0.5

        def sample_trunc_geom(sample_shape=torch.Size()) -> Any:
            # https://stackoverflow.com/questions/16317420/sample-integers-from-truncated-geometric-distribution
            J = int(
                torch.floor(
                    torch.log(1 - torch.rand(sample_shape) * (1 - (1 - p) ** N))
                    / torch.log(1 - p)
                ).item()
            ) + 1
            return J
        
        sample_geom = torch.distributions.Geometric(probs=torch.tensor(p)).sample

        if gen_dist == "geometric":
            self.sample_fn = sample_geom
        elif gen_dist == "trunc_geometric":
            self.sample_fn = sample_trunc_geom
        else:
            raise ValueError(f"Batchsize distribution has to be either \
            geometric ('geometric') or truncated geometric ('trunc_geom'). You've specified {gen_dist}")

    def compute_batchsize(self, seed=None) -> int:
        torch.manual_seed(seed=seed)
        J = self.sample_fn()
        return J

    def _compute_mlmc_estimation(self, closure, batch: tuple, J: int, config: dict):

        J_t = int(torch.ceil((2 ** J) * self._B))
        J_tm1 = int(torch.ceil((2 ** (J - 1)) * self._B))

        #when we compute per-sample gradient estimation, we don't want
        #to call parameters of the graph
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        def compute_loss(batch):
            loss, _ = closure(batch, config)
            return loss
        
        def take_idx(tree: Any, idx: int):
            return map(lambda x: x[idx], tree)  

        def elementwise_sub(tree1: Any, tree2: Any): 
            return map(
                lambda x, y: x - y, tree1, tree2
            )

        if 2**J > self._M:
            # gradients are shaped likewise parameters
            return {
                k: torch.zeros_like(v) for k, v in self.model.named_parameters()
            }
        
        grad_fn = vmap(grad(compute_loss, has_aux=True), in_dims=(0, ), out_dims=(0, 0))
        
        return (2**J_t) * elementwise_sub(
            grad_fn(params, buffers, take_idx(batch, J_t)),
            grad_fn(params, buffers, take_idx(batch, J_tm1))
        )
        
   # this method doesn't modify computational graph
    def compute_loss(self, batch, closure = None, config: dict = {}) -> Any:     
        
        self.sample_fn = self._compute_mlmc_estimation(closure, batch, self.compute_batchsize(), config)

        return closure(*batch)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        def elementwise_add(tree1, tree2):
            return map(
                lambda x, y: x + y, tree1, tree2
            )
        
        for name, param in self.model.named_parameters():
            if self.sample_grads and torch.isfinite(self.sample_grads[name]):
                param.grad = elementwise_add(self.sample_grads[name])
        
        return self.optim.step(closure=closure)