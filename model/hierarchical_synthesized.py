import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
import jax.scipy as jsc
from jax import vmap
class HierarchicalSynthesized:
    N = 1000
    def __init__(self):
        self.y = self.data()
    def data(self):
        key = random.PRNGKey(2)
        res = []
        theta = 4
        for _ in range(self.N):
            key, rng_key = random.split(key)
            x_key, y_key = random.split(rng_key)
            x = dist.Gamma(*self.get_x_parameters(theta)).sample(key=x_key)
            y = dist.Normal(0,1/jnp.sqrt(x)).sample(key=y_key)
            res.append([y])
        return jnp.array(res)
        #print(res, jnp.mean(res), jnp.std(res))

    def model(self):
        theta = numpyro.sample('theta', dist.Gamma(5.,1.))
        x = numpyro.sample('x',dist.Gamma(*self.get_x_parameters(theta)),sample_shape=(self.N,))
        y = numpyro.sample('y', dist.Normal(0, jnp.expand_dims(1/jnp.sqrt(x),axis=1)), obs=self.y)

    def get_x_parameters(self,theta):
        return (theta,theta)

    def prior(self,theta): # For hierarchical models, the prior p(theta) for the unconstrained model should be specified.
        theta_true = jnp.exp(theta)
        theta_dist = dist.Gamma(5.,1.)
        return jnp.sum(theta_dist.log_prob(theta_true) + theta)

    def conditional(self, theta,x,y): # For hierarchical models, the conditional p(x_i,y_i|theta) for each branch in the unconstrained model should be specified. But it will be automated later.
        theta = jnp.exp(theta)
        x_true = jnp.exp(x)
        x_dist = dist.Gamma(*self.get_x_parameters(theta))
        y_dist = dist.Normal(0,1/jnp.sqrt(x_true))
        return jnp.sum(x_dist.log_prob(x_true) + x + y_dist.log_prob(y))

    def set_lambda(self, lamb):
        pass

    def args(self):
        return ()

    def kwargs(self):
        return {}

    def parameters(self):
        return ['theta']

    def name(self):
        return 'HierarchicalSynthesized'

if __name__ == '__main__':
    x = HierarchicalSynthesized()
    print(x.data())