# Multi-level Monte-Carlo (MLMC) and various reinforcement learning algorithms

> Reinforcement learning is wrong on so many levels. 

This repositoru contains code for the paper "Methods for Optimization Problems with Markovian Stochasticity and Non-Euclidean Geometry" and some other snippets that the authors found useful.

## Installation

The environment is built in `docker`, you can build the container:
```Bash
make build
```

And run it in detached mode

```Bash
make run
```

Also, you can just install the requirements for the project with `pip`, `uv`, or conda.

## Experiment scripts

The experiments for the paper are located in the [paper_experiments](./paper_experiments/). Each script is configured with a [hydra](https://hydra.cc) config, entitled the same as the experiment file.

Experiments with projections:
```Bash
cd paper_experiments;
python experiment_mdpo.py
```

Experiments without projections:
```Bash
cd paper_experiments;
python experiment_ampo.py
```

> [!NOTE]
> To enable `wandb` logging, don't forget to specify the `WANDB_API_KEY` environment variable.


