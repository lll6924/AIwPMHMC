import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpyro.distributions as dist
from jax import lax
from numpyro.infer import  SVI, Trace_ELBO
from numpyro.infer.hmc import initialize_model, init_to_uniform
from numpyro.infer.autoguide import AutoIAFNormal
from numpyro.infer.reparam import NeuTraReparam
from numpyro.optim import Adam, SGD,RMSProp
import jax
from jax import jit
class NeuTra:
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
        self.rng_key, train_key, init_key = random.split(rng_key,3)
        self.model_args = args
        self.model_kwargs = kwargs
        self.vi_step_size = vi_step_size
        self.vi_steps = vi_steps


        optimizer = Adam(step_size=vi_step_size)

        guide = AutoIAFNormal(model, num_flows=3)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        init_state = svi.init(train_key, *args, **kwargs)
        # state = lax.fori_loop(0, 100000, lambda i, state: svi.update(state, *model.args(), **model.kwargs())[0], init_state)
        state, losses = lax.scan(lambda state, i: svi.update(state, *args, **kwargs), init_state,
                                 jnp.arange(vi_steps))
        # import matplotlib.pyplot as plt
        # plt.plot(losses)
        # plt.show()
        print(losses)
        params = svi.get_params(state)
        self.neutra = NeuTraReparam(guide, params)

        init_params, potential_fn, self.postprocess_fn, _ = initialize_model(
            init_key,
            self.neutra.reparam(model),
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
        print(self.params)

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
        def neg_log_prob(z):
            log_potential = self.potential_fn(*self.model_args, **self.model_kwargs)(full_translate(z))
            return log_potential

        def postprocess(sites):
            dict = self.neutra.transform_sample(sites)
            return dict
        return n_remained, neg_log_prob, postprocess

    def get_params(self):
        return self.params




