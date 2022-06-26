from approximator import Approximator
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import replay, seed, trace, substitute
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer.svi import SVIState
from jax import lax, vmap
import numpyro
import jax.random as random
from jax.experimental import stax
from numpyro.distributions.flows import (
    InverseAutoregressiveTransform,
)
from numpyro.distributions.transforms import (
    PermuteTransform,
)
from numpyro.infer.util import _get_model_transforms
from numpyro.nn.auto_reg_nn import AutoregressiveNN
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

def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )

class VariationalInference(Approximator):

    def init(self,potential_fn, marginalized, remained, translate, num_sample, *args, rng_key = 0, step_size = 1e-4, steps = 100000, theta_flows = 2, x_flows = 1, hidden_scale = 2, hidden_dim_scale=8, **kwargs):
        self.potential_fn = potential_fn
        self.args = args
        self.kwargs = kwargs
        self.translate = translate
        self.hidden_dim_scale = hidden_dim_scale
        self.hidden_dim = 64
        #self.observation = kwargs['y']
        z_dim = len(marginalized)
        in_dim = len(remained)
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.dim = in_dim+z_dim*num_sample
        rng_key, self.iterate_seed = random.split(rng_key)
        seeds, guide_seed = random.split(rng_key)
        seeds, prior_seed = random.split(seeds)
        def myguide(theta,mu):
            self.encode = numpyro.module("encoder", encoder(self.hidden_dim, z_dim), (in_dim ,))
            z_loc, z_std = self.encode(theta)
            cond_dist = dist.Normal(loc=z_loc, scale=z_std)
            z_loc = jnp.expand_dims(z_loc, 1)
            z_std = jnp.expand_dims(z_std, 1)
            base = z_loc + z_std * mu
            base = base.transpose()
            log_prob = jnp.sum(cond_dist.log_prob(base), axis=1)
            return base, log_prob


        optimizer = numpyro.optim.Adam(step_size=step_size)

        seeds, guide_seed = random.split(seeds)
        guide_init = seed(myguide, guide_seed)
        guide_trace = trace(guide_init).get_trace(jnp.zeros(in_dim),jnp.zeros((self.z_dim,num_sample)))
        params = {}
        inv_transforms = {}

        for site in list(guide_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        constrain_fn = partial(transform_fn, inv_transforms)
        seeds, state_seed = random.split(seeds)
        init_loc = jnp.zeros(self.dim)
        init_scale = jnp.zeros(self.dim)
        #scale_tril = numpyro.param(
        #    "scale_tril",
        #    jnp.identity(self.dim),
        #    constraint=constraints.scaled_unit_lower_cholesky,
        #)
        #print(scale_tril)
        init_state = SVIState(optimizer.init((params,init_loc,init_scale)), None, state_seed)
        print(self.in_dim, self.z_dim)
        #print(init_state)
        def update(svi_state, optimizer, *args, **kwargs):
            rng_key, rng_key_step = random.split(svi_state.rng_key)
            def loss_fn(params):
                param, loc, scale = params
                param = constrain_fn(param)
                guide = substitute(myguide, data=param)
                z_base_dist = dist.Normal(loc, jnp.exp(scale))
                z = numpyro.sample('z_base', z_base_dist, rng_key=rng_key_step)
                logq = jnp.sum(z_base_dist.log_prob(z))
                z1 = z[:self.in_dim]
                z2 = jnp.reshape(z[self.in_dim:], (self.z_dim,num_sample))
                samples, qlogpdf = guide(z1, z2)

                def single_sample_logpdf(z2):
                    dict = translate(z1,z2)
                    return -self.potential_fn(*args, **kwargs)(dict)

                logpdfs = vmap(single_sample_logpdf)(samples)
                logpdfs = jnp.asarray(logpdfs)
                density = logmeanexp(logpdfs - qlogpdf)+jnp.sum(dist.Normal(0, 1).log_prob(z2))

                return (logq-density,None)
            #loss_fn(optimizer.get_params(init_state.optim_state))
            (loss_val, mutable_state), optim_state = optimizer.eval_and_update(
                loss_fn, svi_state.optim_state
            )
            return SVIState(optim_state, None, rng_key), loss_val


        #update(init_state, *args, **kwargs)
        losses1 = jnp.zeros(steps)

        def body(i, states):
            state, losses = states
            newstate, loss = update(state, optimizer, *args, **kwargs)
            return (newstate,losses.at[i].set(loss))

        state, losses1 = lax.fori_loop(0, steps, body, (init_state, losses1))
        print(losses1)

        # optimizer2 = numpyro.optim.Adam(step_size=step_size/10)
        # def body2(i, states):
        #     state, losses = states
        #     newstate, loss = update(state, optimizer2, *args, **kwargs)
        #     return (newstate,losses.at[i].set(loss))
        #
        # state, losses1 = lax.fori_loop(0, steps, body2, (state, losses1))
        # print(losses1)
        #
        # optimizer3 = numpyro.optim.Adam(step_size=step_size/100)
        # def body3(i, states):
        #     state, losses = states
        #     newstate, loss = update(state, optimizer3, *args, **kwargs)
        #     return (newstate,losses.at[i].set(loss))
        #
        # state, losses1 = lax.fori_loop(0, steps, body3, (state, losses1))
        # print(losses1)


        datas = []
        for i in range(steps):
            if(i%100==0):
                datas.append({'x': i, 'y': float(losses1[i])})

        #datas = pd.DataFrame(datas)
        #sns.lineplot(data=datas, x='x', y='y')
        #plt.ylim([500, 600])
        #plt.show()

        self.state = state
        self.params, loc, scale = optimizer.get_params(state.optim_state)
        print(loc,scale)
        self.params = constrain_fn(self.params)

        self.guide = substitute(myguide, data=self.params)






    def apply(self, theta, mu):
        mu = mu.transpose()
        base, log_prob = self.guide(theta,mu)

        return base, log_prob
