import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpyro.distributions as dist
from numpyro.infer.hmc import initialize_model, init_to_uniform
from approximator import VariationallyInferredParameterization
from utils.extensions import log_mean_exp
import sys
import time
import jax
from jax import jit
class VIP:
    def __init__(self,
                 model,
                 rng_key,
                 *args,
                 init_strategy=init_to_uniform,
                 vi_step_size = 1e-4,
                 vi_steps = 100000,
                 record = False,
                 model_class=None,
                 **kwargs):
        """
        :param model: The model to infer, the same as what is passed to numpyro.infer.MCMCKernel
        :param rng_key:
        :param args: The arguments of the model
        :param num_sample: Number of samples for PMHMC
        :param init_strategy: The initialize function passed to numpyro.infer.utils.initialize_model
        :param approximate_strategy: 'VI' for Variational Inference based approximation and 'LA' for Laplace's approximation
        :param vi_step_size: The step size for each step of 'VI'
        :param vi_steps: The number of steps for 'VI'
        :param kwargs: The (key = word) arguments of the model
        """
        self.model = model
        self.rng_key, init_key = random.split(rng_key)
        self.model_args = args
        self.model_kwargs = kwargs
        self.vi_step_size = vi_step_size
        self.vi_steps = vi_steps
        init_params, potential_fn, self.postprocess_fn, _ = initialize_model(
            init_key,
            model,
            dynamic_args=True,
            init_strategy=init_strategy,
            model_args=args,
            model_kwargs = kwargs)
        self.params = sorted(list(init_params.z.keys()))
        self.initials = init_params.z

        #self.potential_fn_evaluations = jnp.asarray([0])
        #self.potential_fn_time = jnp.asarray([0])
        self.record = record
        if record:
            @jax.profiler.annotate_function
            def potential(*args, **kwargs):
                #start = time.time_ns()
                #self.potential_fn_evaluations += 1
                res = potential_fn(*args, **kwargs)
                #end = time.time_ns()
                #self.potential_fn_time += end - start
                return res
            self.potential_fn = potential
        else:
            self.potential_fn = potential_fn

        # preprocess the information of parameters in the model
        self.par_dims = []
        self.par_shapes = []
        for par in self.params:
            if init_params.z[par].shape == ():
                self.par_dims.append(jnp.asarray(1))
                self.par_shapes.append((1,))
            else:
                self.par_dims.append(jnp.prod(jnp.asarray(init_params.z[par].shape)))
                self.par_shapes.append(init_params.z[par].shape)
        self.n_dims = jnp.sum(jnp.asarray(self.par_dims))
        self.n_params = len(self.params)
        self.vi_step_size = vi_step_size
        self.vi_steps = vi_steps

    def compile(self,
            par_list=None,
            num_warmup=6000,
            num_samples=6000):
        """
        :param par_list: The name of parameters for PMHMC to infer
        :param num_warmup: The number of warm-up steps
        :param num_samples: The number of sampling steps
        :return:
        """

        if par_list is None:
            par_list = self.params
        for par in par_list:
            assert(par in self.params)

        # preprocess the id, range of two sets of parameters.
        # `marginalize' represents marginalized parameters
        # `remain' represents parameters in par_list
        n_remained = 0
        remained = []
        remain_id = []
        remain_l = []
        remain_r = []
        initials = jnp.array([])
        now = 0

        for i,dim,par in zip(range(self.n_params),self.par_dims, self.params):
            remained.extend(range(now,now+dim))
            remain_l.append(n_remained)
            remain_r.append(n_remained+dim)
            n_remained += dim
            remain_id.append(i)
            initials = jnp.append(initials, jnp.reshape(self.initials[par],(-1)))
            now += dim
        def full_translate(z):
            dict = {}
            for i in range(len(remain_id)):
                _i = remain_id[i]
                dict[self.params[_i]] = jnp.reshape(z[remain_l[i]:remain_r[i]], self.par_shapes[_i])
            return dict

        self.n_remained = n_remained
        self.remain_id = remain_id
        self.remain_l = remain_l
        self.remain_r = remain_r
        # get the approximation of posterior distribution
        self.rng_key, approximator_key = random.split(self.rng_key)
        self.rng_key, approximator_key = random.split(self.rng_key)
        self.approximator = VariationallyInferredParameterization()
        lamb = self.approximator.init(self.potential_fn, remained, full_translate, *self.model_args,
                               rng_key=approximator_key, step_size=self.vi_step_size, steps=self.vi_steps,
                               **self.model_kwargs)
        self.model_kwargs['lamb'] = lamb
        def neg_log_prob(z):
            log_potential = self.potential_fn(*self.model_args, **self.model_kwargs)(full_translate(z))
            return log_potential

        def postprocess(sites):
            ret = {}
            for i in range(len(remain_id)):
                _i = remain_id[i]
                ret[self.params[_i]] = sites[:, :, remain_l[i]:remain_r[i]]
                ret[self.params[_i]] = jnp.reshape(ret[self.params[_i]],
                                                   ret[self.params[_i]].shape[:2] + self.par_shapes[_i])

            # postprocess the chains to match the support of original distributions
            ret = self.postprocess_fn(*self.model_args, **self.model_kwargs)(ret)

            # if self.record:
            #    import numpy as np
            #    print((self.potential_fn_time[0]))
            #    print((self.potential_fn_evaluations[0]))

            return ret#dict(filter(lambda kv: kv[0] in par_list, ret.items()))
        return n_remained, neg_log_prob, postprocess

    def get_params(self):
        return self.params




