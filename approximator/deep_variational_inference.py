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
def getflows(num_flows,z_dim, hidden_scale,name):
    flows = []
    for i in range(num_flows):
        if i > 0:
            flows.append(PermuteTransform(jnp.arange(z_dim)[::-1]))
        arn = AutoregressiveNN(
            z_dim,
            [z_dim*hidden_scale, z_dim*hidden_scale],
            permutation=jnp.arange(z_dim),
            nonlinearity=stax.Elu,
        )
        arnn = numpyro.module(
            "{}_arn_{}_{}".format('nf',name, i), arn, (z_dim,)
        )
        flows.append(InverseAutoregressiveTransform(arnn))
    return flows

class DeepVariationalInference(Approximator):

    def init(self,potential_fn, marginalized, remained, translate, num_sample, *args, rng_key = 0, step_size = 1e-4, steps = 100000, theta_flows = 2, x_flows = 1, hidden_scale = 2, hidden_dim_scale=6, **kwargs):
        z_dim = len(marginalized)
        in_dim = len(remained)
        self.hidden_dims = 64
        num_sample = 1 # training VI instead of IWVI
        seeds, guide_seed = random.split(rng_key)

        def myguide(theta):
            self.encode = numpyro.module("encoder", encoder(self.hidden_dims, z_dim), (in_dim,))
            #self.flows = getflows(x_flows,z_dim,hidden_scale,'z')
            z_loc, z_std = self.encode(theta)
            cond_dist = dist.Normal(loc = z_loc, scale = z_std)
            #flow_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_std), self.flows)
            proposed = numpyro.sample('z', cond_dist,sample_shape = (num_sample,))
            return proposed, jnp.sum(cond_dist.log_prob(proposed),axis=1)


        optimizer = numpyro.optim.Adam(step_size=step_size)

        seeds, guide_seed = random.split(seeds)
        guide_init = seed(myguide, guide_seed)
        guide_trace = trace(guide_init).get_trace(jnp.zeros(in_dim))
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
        init_state = SVIState(optimizer.init((params, jnp.zeros(in_dim), jnp.zeros(in_dim))), None, state_seed)
        #print(init_state)
        def update(svi_state, optimizer, *args, **kwargs):
            rng_key, rng_key_step = random.split(svi_state.rng_key)

            def loss_fn(params):
                sampling_key, seed_key = random.split(rng_key_step)
                params, theta_loc, theta_scale = params
                params = constrain_fn(params)
                guide = substitute(myguide, data=params)
                guide_init = seed(guide, seed_key)
                theta_dist = dist.Normal(theta_loc, jnp.exp(theta_scale))
                theta = theta_dist.sample(sampling_key)
                proposed, logq = guide_init(theta)
                logq = logq + theta_dist.log_prob(theta)
                def getp(proposed):

                    dict = translate(theta, proposed)
                    logp = - potential_fn(*args, **kwargs)(dict)
                    return logp
                logps = vmap(getp)(proposed)
                return (-logmeanexp(logps-logq), None,)
            #loss_fn(optimizer.get_params(init_state.optim_state))
            (loss_val, mutable_state), optim_state = optimizer.eval_and_update(
                loss_fn, svi_state.optim_state
            )
            return SVIState(optim_state, mutable_state, rng_key), loss_val
        #update(init_state, *args, **kwargs)
        losses1 = jnp.zeros(steps)

        def body(i, states):
            state, losses = states
            newstate, loss = update(state, optimizer, *args, **kwargs)
            return (newstate, losses.at[i].set(loss))
        state, losses1 = lax.fori_loop(0, steps, body, (init_state, losses1))
        def plotting():
            datas = []
            for i in range(steps):
                datas.append({'x':i,'y':float(losses1[i])})

            datas = pd.DataFrame(datas)
            sns.lineplot(data=datas,x='x',y='y')
            plt.ylim([0,100])
            plt.show()
            print(type(state),len(state))
        #plotting()
        print(losses1)
        self.params, _, _ = optimizer.get_params(state.optim_state)
        self.params = constrain_fn(self.params)
        def newguide(theta,mu):
            self.encode = numpyro.module("encoder", encoder(self.hidden_dims, z_dim), (in_dim,))
            #self.flows = getflows(x_flows,z_dim, hidden_scale,'z')
            z_loc, z_std = self.encode(theta)
            #flow_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_std), self.flows)
            cond_dist = dist.Normal(loc = z_loc, scale = z_std)
            #transform = ComposeTransform(self.flows)
            z_loc = jnp.expand_dims(z_loc,1)
            z_std = jnp.expand_dims(z_std,1)
            base = z_loc + z_std * mu
            base = base.transpose()
            #base = transform(base)
            log_prob = jnp.sum(cond_dist.log_prob(base),axis=1)
            return base, log_prob


        def plotguide(rng_key):
            self.theta_flow = getflows(theta_flows,in_dim,hidden_scale,'theta')
            theta_dist = dist.TransformedDistribution(dist.Normal(jnp.zeros(in_dim), jnp.ones(in_dim)), self.theta_flow)
            return numpyro.sample('theta', theta_dist,rng_key=rng_key)
        self.plotguide = substitute(plotguide, data=self.params)
        def plotg():
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            data = []
            rng_key = random.PRNGKey(0)
            for _ in range(50000):
                rng_key, sample_key = random.split(rng_key)
                sample = self.plotguide(rng_key = sample_key)
                #print(sample)
                data.append({'mu': float(sample[0]), 'logtau':float(sample[1])})
            data = pd.DataFrame(data)
            sns.scatterplot(x="mu", y='logtau', data = data, s=5, color=".15")
            sns.histplot(x="mu", y='logtau', data = data, bins=50, pthresh=.05, cmap="mako")
            sns.kdeplot(x="mu", y='logtau', data = data, levels=5, color="w", linewidths=1)
            plt.xlim([-10,15])
            plt.ylim([-8,4])
            plt.show()
        # plotg()
        def preconditioner(theta_base):
            self.theta_flow = getflows(theta_flows,in_dim, hidden_scale,'theta')
            transform = ComposeTransform(self.theta_flow)
            theta = transform(theta_base)
            return theta, transform.log_abs_det_jacobian(theta_base,theta)

        self.guide = substitute(newguide, data=self.params)
        #self.preconditioner = substitute(preconditioner, data = self.params)

        def postprocess(sites):
            sites = sites[..., :in_dim]
            sites, _ = preconditioner(sites)
            return sites
        #self.postprocess = postprocess

    def apply(self, theta, mu):
        mu = mu.transpose()
        return self.guide(theta,mu)