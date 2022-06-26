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

class VariationalInferenceHierarchical(Approximator):

    def init(self,model_class, marginalized, remained, translate, num_sample, *args, rng_key = 0, step_size = 1e-4, steps = 100000, theta_flows = 2, x_flows = 1, hidden_scale = 2, hidden_dim_scale=8, quadrature = False, **kwargs):
        real_num_sample = num_sample
        num_sample = 1

        self.prior = model_class.prior
        self.conditional = model_class.conditional
        self.args = args
        self.kwargs = kwargs
        self.translate = translate
        self.hidden_dim_scale = hidden_dim_scale
        self.hidden_dim = 16
        self.observation = model_class.y
        self.N = len(self.observation)
        self.y_dim = len(self.observation[0])
        z_dim = len(marginalized)//self.N
        in_dim = len(remained)+self.y_dim
        self.z_dim = z_dim
        self.in_dim = in_dim
        self.dim = in_dim+z_dim*num_sample
        rng_key, self.iterate_seed = random.split(rng_key)
        seeds, guide_seed = random.split(rng_key)
        seeds, prior_seed = random.split(seeds)

        def myguide(theta, mu):
            self.encode = numpyro.module("encoder", encoder(self.hidden_dim, z_dim), (in_dim,))
            theta_all = jnp.repeat(jnp.array([theta]),self.N,axis=0)
            z_loc, z_std = self.encode(jnp.concatenate([theta_all,self.observation],axis=1))
            z_loc = jnp.expand_dims(z_loc, 1)
            z_std = jnp.expand_dims(z_std, 1)
            cond_dist = dist.Normal(loc=z_loc, scale=z_std)
            mu = jnp.reshape(mu,(self.N,z_dim,num_sample,))
            mu = jnp.transpose(mu,axes=[0,2,1])

            base = z_loc + z_std * mu
            log_prob = jnp.sum(cond_dist.log_prob(base), axis=2)
            return base, log_prob


        optimizer = numpyro.optim.Adam(step_size=step_size)

        seeds, guide_seed = random.split(seeds)
        guide_init = seed(myguide, guide_seed)
        guide_trace = trace(guide_init).get_trace(jnp.zeros(len(remained)),jnp.zeros((self.z_dim*self.N,num_sample)))
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
        init_state = SVIState(optimizer.init((params,jnp.zeros(len(remained)),jnp.zeros(len(remained)))), None, state_seed)
        #print(init_state)
        def update(svi_state, optimizer, *args, **kwargs):
            rng_key, rng_key1, rng_key2 = random.split(svi_state.rng_key, 3)

            def loss_fn(params):
                param, theta_loc, theta_scale = params
                theta_scale = jnp.exp(theta_scale)
                param = constrain_fn(param)
                guide = substitute(myguide, data=param)

                mu_dist = dist.Normal(0, 1)
                theta_dist = dist.Normal(theta_loc, theta_scale)
                mu = numpyro.sample('z_base', mu_dist, rng_key=rng_key2,
                                    sample_shape=(self.z_dim * self.N, num_sample))
                theta = numpyro.sample('theta', theta_dist, rng_key=rng_key1)
                base, log_prob = guide(theta, mu)

                def single_z_logpdf(xs, ys, logqs):
                    def single_sample_logpdf(x, y):
                        return self.conditional(theta, x, y)

                    logps = vmap(single_sample_logpdf)(xs, ys)
                    return logmeanexp(logps - logqs)

                logpdfs = vmap(single_z_logpdf)(base,
                                                jnp.repeat(jnp.expand_dims(self.observation, axis=1), num_sample,
                                                           axis=1), log_prob)

                return (-jnp.sum(logpdfs) - self.prior(theta) + jnp.sum(theta_dist.log_prob(theta)), None)

            # loss_fn(optimizer.get_params(init_state.optim_state))
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

        def pmhmc_guide(theta,mu):
            self.encode = numpyro.module("encoder", encoder(self.hidden_dim, z_dim), (in_dim,))
            theta_all = jnp.repeat(jnp.array([theta]),self.N,axis=0)
            z_loc, z_std = self.encode(jnp.concatenate([theta_all,self.observation],axis=1))
            z_loc = jnp.expand_dims(z_loc, 1)
            z_std = jnp.expand_dims(z_std, 1)
            cond_dist = dist.Normal(loc=z_loc, scale=z_std)
            mu = jnp.reshape(mu,(self.N,z_dim,real_num_sample,))
            mu = jnp.transpose(mu,axes=[0,2,1])

            base = z_loc + z_std * mu
            log_prob = jnp.sum(cond_dist.log_prob(base), axis=2)
            return base, log_prob

        self.state = state
        self.params,theta_loc,_ = optimizer.get_params(state.optim_state)
        self.params = constrain_fn(self.params)
        self.guide = substitute(pmhmc_guide, data=self.params)





    def apply(self, theta, mu):
        mu = mu.transpose()
        base, log_prob = self.guide(theta,mu)

        return base, log_prob
