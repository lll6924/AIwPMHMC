import jax.numpy as jnp

class Approximator:

    def apply(self, theta, mu):
        raise NotImplementedError()

    def preprocess(self, theta):
        return theta,jnp.asarray(0)

    def postprocess(self,sites):
        return sites

    def iterate(self, samples, seeds=None):
        pass