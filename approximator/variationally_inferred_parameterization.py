from approximator import Approximator
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import replay, seed, trace, substitute
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer.svi import SVIState
from jax import lax, vmap
import numpyro
import jax.random as random
import jax.scipy as jscipy
from functools import partial
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import transform_fn
from numpyro.distributions import constraints
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def logmeanexp(x):
    m = jnp.max(x)
    return m + jnp.log(jnp.mean(jnp.exp(x-m)))

class VariationallyInferredParameterization(Approximator):

    def init(self,potential_fn, remained, translate, *args, rng_key = 0, step_size = 1e-4, steps = 100000, **kwargs):
        self.potential_fn = potential_fn
        self.args = args
        self.kwargs = kwargs
        self.translate = translate
        all_dim = len(remained)
        #print(remained, marginalized)
        self.all_dim = all_dim
        optimizer = numpyro.optim.RMSProp(step_size=step_size)

        seeds, state_seed = random.split(rng_key)
        init_loc = jnp.zeros(all_dim)
        init_scale = jnp.ones(all_dim)
        init_lambda = kwargs['lamb']
        init_state = SVIState(optimizer.init((init_loc, init_scale, init_lambda)), None, state_seed)
        #print(init_state)
        def update(svi_state, optimizer, *args, **kwargs):
            rng_key, rng_key_step = random.split(svi_state.rng_key)
            def loss_fn(params):
                loc, scale, lamb = params
                z_base_dist = dist.Normal(loc, scale)
                z = numpyro.sample('z_base',z_base_dist, rng_key=rng_key_step)
                logq = jnp.sum(z_base_dist.log_prob(z))
                the_kwargs = kwargs.copy()
                the_kwargs['lamb'] = lamb
                dict = translate(z)
                logp = - potential_fn(*args, **the_kwargs)(dict)
                return (logq-logp, logp-logq,)
            #loss_fn(optimizer.get_params(init_state.optim_state))
            (loss_val, mutable_state), optim_state = optimizer.eval_and_update(
                loss_fn, svi_state.optim_state
            )
            return SVIState(optim_state, None, rng_key), loss_val, mutable_state

        #update(init_state, *args, **kwargs)
        losses1 = jnp.zeros(steps)

        def body(i, states):
            state, losses = states
            newstate, loss, loglogs = update(state, optimizer, *args, **kwargs)
            return (newstate, losses.at[i].set(loss))

        state, losses1 = lax.fori_loop(0, steps, body, (init_state, losses1))
        print(losses1)
        datas = []
        for i in range(steps):
            if(i%100==0):
                datas.append({'x': i, 'y': float(losses1[i])})

        #datas = pd.DataFrame(datas)
        #sns.lineplot(data=datas, x='x', y='y')
        #plt.ylim([500, 600])
        #plt.show()

        self.state = state
        loc, scale, lamb = optimizer.get_params(state.optim_state)
        self.loc = loc
        self.scale = scale
        self.lamb = lamb
        print(lamb)
        return self.lamb
