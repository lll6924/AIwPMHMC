import jax.numpy as jnp

def log_mean_exp(x):
    mx = jnp.max(x)
    return jnp.log(jnp.mean(jnp.exp(x-mx)))+mx

