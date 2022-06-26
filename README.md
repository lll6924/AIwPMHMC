# Automatic Inference with Pseudo-Marginal Hamiltonian Monte Carlo

Accepted to ICML workshop, Beyond Bayes: Paths Towards Universal Reasoning Systems. The codes are under update. 

## Model Definition

We define the models using NumPyro's codes. See `model/gamma_funnel.py` as an example. Inside the class of the model, the following functions should be specified:

* `model`: the NumPyro codes to define the model;
* `set_lambda`: unconstrained reparameterization factor in the model, interpolating the model between non-centered parameterization and centered parameterization. Note that the model should be implemented accordingly. See `eight_schools.py` for an example;
* `args`: the arguments feeded into the model;
* `kwargs`: the key value pairs feeded into the model;
* `parameters`: the specified $\theta$ for PM-HMC.

## Running PM-HMC

To run PM-HMC with leapfrog integrators, the following scripts should be used:

``
python -m test.run_hmc --algorithm PMHMC2 --model $YourModel
``

To run PM-HMC with operator splitting, the following scripts should be used:

``
python -m test.run --model $YourModel
``

## Running other algorithms

To run vanilla HMC, use 

``
python -m test.run_hmc --model $YourModel
``

To run NeuTra, use

``
python -m test.run_hmc --algorithm NeuTra --model $YourModel
``

To run VIP, use

``
python -m test.run_hmc --algorithm VIP --approximator VariationallyInferredParameterization --model $YourModel
``

## Other Attributes

The algorithms are also controlled by the following attributes:

* `--warmup_steps`: the warm up steps before sampling;
* `--sample_steps`: the number of samples in HMC;
* `--rng_key`: the random seed;
* `--rho`: the $\rho$ in the informative eight schools model;
* `--trajectory_length`: the trajectory length for each step in HMC;
* `--training_steps`: the training steps if the algorithm involves VI;
* `--training_lr`: the training learning rate for VI;
* `--lamb`: the reparameterization factor in the model ($-\infty$ for non-centered parameterization and $\infty$ for centered parameterization, practically setting between -10 and 10 is enough);
* `--samples`: the number of samples for PM-HMC.

## Hierarchical Models;

Updating now...