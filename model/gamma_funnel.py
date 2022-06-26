import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

class GammaFunnel:
    def model(self):
        logbeta = numpyro.sample('beta', dist.Normal(0,1))
        with numpyro.plate("num_samples", 9):
            x = numpyro.sample('x', dist.Gamma(jnp.exp(logbeta),jnp.exp(logbeta)))

    def set_lambda(self, lamb):
        pass
    def args(self):
        return ()

    def kwargs(self):
        return {}

    def parameters(self):
        return ['beta']