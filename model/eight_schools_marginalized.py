import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
import jax.numpy as jnp

class EightSchoolsMarginalized:
    J = 8
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    # Eight Schools example
    def model(self, J, sigma, y):
        mu = numpyro.sample('mu', dist.Normal(0, 5))
        tau = numpyro.sample('tau', dist.HalfCauchy(5))
        with numpyro.plate('J', J):
            numpyro.sample('obs', dist.Normal(mu, jnp.sqrt(jnp.square(tau)+jnp.square(sigma))), obs=y)

    def set_parameters(self, J,rho):
        self.J = J
        self.y = self.y[:J]
        self.sigma = self.sigma[:J]*rho

    def set_lambda(self, lamb):
        self.lamb_0 = lamb
    def args(self):
        return (self.J,self.sigma)

    def kwargs(self):
        return {'y':self.y}

    def parameters(self):
        return ['mu','tau']