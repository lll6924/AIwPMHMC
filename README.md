# Automatic Inference with Pseudo-Marginal Hamiltonian Monte Carlo

Accepted to ICML workshop, Beyond Bayes: Paths Towards Universal Reasoning Systems. The codes are under update. 

## Model Definition

We define the models using NumPyro's codes. See `model/gamma_funnel.py` as an example. Inside the class of the model, the following functions should be specified:

* `model`: the NumPyro codes to define the model;
* `set_lambda`: unconstrained reparameterization factor in the model, interpolating the model between non-centered parameterization and centered parameterization. Note that the model should be implemented accordingly. See `eight_schools.py` for an example;
* `args`: the arguments feeded into the model;
* `kwargs`: the key value pairs feeded into the model;
* `parameters`: the specified $$\theta$$ for PM-HMC.

## Running

To run with leapfrog integrators, the following scripts should be used:

``
python -m test.run_hmc 
``

To run with operator splitting, the following scripts should be used:

``
python -m test.run
``