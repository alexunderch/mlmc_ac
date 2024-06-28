# Applying Multi-level Monte-Carlo (MLMC) to various reinforcement learning algorithms

Or reinforcement learning is wrong on so many levels. 

Majority of deep existing reinforcement learning methods utilise stochatic optimisation via backpropagation as their general solution back end. However, many of modern optimisation methods assume that data samples are identically distributed, or a distribution of the underlying Markov chain mixes exponentially fast with every parameters update. Notably, the assumption is often violated for Markov decision processes (MDPs) with high-dimensional state spaces or cases when ones have sparse rewards --- i.e. when mixing time could not be exactly estimated or even may be unknown. Fortunately, multi-level Monte Carlo methods, taking into account nature of Markov Chains and letting control variance of the updates, have recently been popularised in the field. They allow to get reasonable gradient estimations even for aforementioned MDPs without explicit knowledge of one's mixing time. We propose a general optimisation framework incorporating multi-level gradient estimators with randomised batch size in methods for discounted and continual reinforcement learning algorithms and analyse performance for different set of environments. We hope that our work will highlight importance not only of reinforcement algorithms but more deliberate choice of optimisation procedures.

Repository structure
```Bash
.
|-- actor_critic
|   |-- continual_ac
|   |   |-- config.yaml
|   |   |-- experiment.py
|   |   |-- sweep_config.yaml
|   |-- continual_ppo
|   |   |-- config.yaml
|   |   |-- experiment.py
|   |   |-- sweep_config.yaml
|   |   |-- sweep_config_batch.yaml
|   |-- continual_td0
|   |   |-- config.yaml
|   |   |-- experiment.py
|   |   |-- sweep_config.yaml
|   |-- discounted_ac
|   |   |-- config.yaml
|   |   |-- experiment.py
|   |   `-- sweep_config.yaml
|   `-- discounted_ppo
|       |-- config.yaml
|       |-- experiment.py
|       |-- sweep_config.yaml
|       |-- sweep_config_batch.yaml
|-- reinforce
|   |-- continual
|   |   |-- config.yaml
|   |   |-- experiment.py
|   |   |-- sweep_config.yaml
|   `-- discounted
|       |-- config.yaml
|       |-- experiment.py
|       `-- sweep_config.yaml
`-- requirements.txt
```

