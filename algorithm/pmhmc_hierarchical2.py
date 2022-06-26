import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
import numpyro.distributions as dist
from numpyro.infer.hmc import initialize_model, init_to_uniform
from approximator import  VariationalInferenceHierarchical
from utils.extensions import log_mean_exp
from jax.scipy.special import logsumexp
import sys
import time
import jax
from jax import jit
class PMHMCH2:
    def __init__(self,
                 model,
                 prior,
                 conditional,
                 rng_key,
                 *args,
                 num_sample = 20,
                 init_strategy=init_to_uniform,
                 approximate_strategy = 'DeepVariationalInference',
                 vi_step_size = 1e-4,
                 vi_steps = 100000,
                 record = False,
                 model_class = None,
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
        self.prior = prior
        self.conditional = conditional
        self.approximate_strategy = approximate_strategy
        self.num_sample = num_sample
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
        self.approximate_strategy = approximate_strategy
        self.vi_step_size = vi_step_size
        self.vi_steps = vi_steps
        self.model_class = model_class

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
        n_marginalized = 0
        marginalized = []
        remained = []
        marginalize_id = []
        remain_id = []
        remain_l = []
        marginalize_l = []
        remain_r = []
        marginalize_r = []
        initials = jnp.array([])
        now = 0

        for i,dim,par in zip(range(self.n_params),self.par_dims, self.params):
            if par in par_list:
                remained.extend(range(now,now+dim))
                remain_l.append(n_remained)
                remain_r.append(n_remained+dim)
                n_remained += dim
                remain_id.append(i)
                initials = jnp.append(initials, jnp.reshape(self.initials[par],(-1)))
            else:
                marginalized.extend(range(now,now+dim))
                marginalize_l.append(n_marginalized)
                marginalize_r.append(n_marginalized + dim)
                n_marginalized += dim
                marginalize_id.append(i)
            now += dim
        def full_translate(z1, z2):
            dict = {}
            for i in range(len(remain_id)):
                _i = remain_id[i]
                dict[self.params[_i]] = jnp.reshape(z1[remain_l[i]:remain_r[i]], self.par_shapes[_i])

            for i in range(len(marginalize_id)):
                _i = marginalize_id[i]
                dict[self.params[_i]] = jnp.reshape(z2[marginalize_l[i]:marginalize_r[i]],
                                                    self.par_shapes[_i])
            return dict

        def prior_translate(z1):
            dict = {}
            for i in range(len(remain_id)):
                _i = remain_id[i]
                dict[self.params[_i]] = jnp.reshape(z1[remain_l[i]:remain_r[i]], self.par_shapes[_i])
            return dict

        def prior_extract(dict):
            ret = jnp.array([])
            for i in range(len(remain_id)):
                _i = remain_id[i]
                ret = jnp.append(ret,dict[self.params[_i]].flatten())
            return ret

        self.n_remained = n_remained
        self.n_marginalized = n_marginalized
        self.remain_id = remain_id
        self.marginalize_id = marginalize_id
        self.remain_l = remain_l
        self.remain_r = remain_r
        self.marginalize_l = marginalize_l
        self.marginalize_r = marginalize_r
        # get the approximation of posterior distribution
        self.rng_key, approximator_key = random.split(self.rng_key)
        self.approximator = getattr(sys.modules[__name__],self.approximate_strategy)()
        if self.approximate_strategy =='VariationalInferenceHierarchical':
            self.approximator.init(self.model_class, marginalized, remained, full_translate, self.num_sample, *self.model_args, rng_key=approximator_key, step_size=self.vi_step_size, steps=self.vi_steps, **self.model_kwargs)
        else:
            raise ValueError('Invalid approximate strategy: {}'.format(str(self.approximate_strategy)))
        def neg_log_prob(z):
            z1 = z[:n_remained]
            z1, cond_logdet = self.approximator.preprocess(z1)
            z2 = jnp.reshape(z[n_remained:], (self.num_sample,n_marginalized))
            samples, qlogpdf = self.approximator.apply(z1,z2)

            def single_z_logpdf(xs, ys, logqs):
                def single_sample_logpdf(x, y):
                    return self.conditional(z1, x, y)

                logps = vmap(single_sample_logpdf)(xs, ys)
                return log_mean_exp(logps - logqs)

            logpdfs = vmap(single_z_logpdf)(samples,jnp.repeat(jnp.expand_dims(self.model_class.y,axis=1),self.num_sample,axis=1),qlogpdf)

            return (- jnp.sum(logpdfs) - self.model_class.prior(z1)  - cond_logdet) - jnp.sum(dist.Normal(0,1).log_prob(z2))
        def postprocess(sites, zs, key):
            retained = []
            def get_sample(theta,z,rng_key):
                theta = jnp.array(theta)
                z = jnp.array(z)
                theta, _ = self.approximator.preprocess(theta)
                z = jnp.reshape(z, (self.num_sample, n_marginalized))
                samples, qlogpdf = self.approximator.apply(theta, z)

                def single_z_logpdf(xs, ys, logqs):
                    def single_sample_logpdf(x, y):
                        return self.conditional(theta, x, y)

                    logps = vmap(single_sample_logpdf)(xs, ys)
                    return logps - logqs

                logpdfs = vmap(single_z_logpdf)(samples,
                                                jnp.repeat(jnp.expand_dims(self.model_class.y, axis=1), self.num_sample,
                                                           axis=1), qlogpdf)
                #print(logpdfs.shape)
                id = random.categorical(rng_key, logpdfs,axis=1)
                def indexing(a,b):
                    return a[b]
                res = vmap(indexing)(samples,id)
                return res
            get_sample_compiled = jit(get_sample)
            for theta,z in zip(sites,zs):
                key, rng_key = random.split(key)
                retained.append(get_sample_compiled(theta,z,rng_key))



            sites = self.approximator.postprocess(sites)
            retained = np.array(retained)

            ret = {}
            for i in range(len(remain_id)):
                _i = remain_id[i]
                ret[self.params[_i]] = sites[:, remain_l[i]:remain_r[i]]
                ret[self.params[_i]] = jnp.reshape(ret[self.params[_i]],
                                                   (1,)+ret[self.params[_i]].shape[:1] + self.par_shapes[_i])
            for i in range(len(marginalize_id)):
                _i = marginalize_id[i]
                ret[self.params[_i]] = retained[:, marginalize_l[i]:marginalize_r[i]]
                ret[self.params[_i]] = jnp.reshape(ret[self.params[_i]],
                                                   (1,) + ret[self.params[_i]].shape[:1] + self.par_shapes[_i])

            # postprocess the chains to match the support of original distributions
            ret = self.postprocess_fn(*self.model_args, **self.model_kwargs)(ret)

            # if self.record:
            #    import numpy as np
            #    print((self.potential_fn_time[0]))
            #    print((self.potential_fn_evaluations[0]))

            return ret #dict(filter(lambda kv: kv[0] in par_list, ret.items()))
        return n_remained, n_marginalized, neg_log_prob, postprocess, self.approximator.iterate, self.recompile

    def recompile(self):
        def neg_log_prob(z):
            z1 = z[:self.n_remained]
            z1, cond_logdet = self.approximator.preprocess(z1)
            z2 = jnp.reshape(z[self.n_remained:], (self.num_sample,self.n_marginalized))
            samples, qlogpdf = self.approximator.apply(z1,z2)
            unchanged_dict = {}
            for i in range(len(self.remain_id)):
                _i = self.remain_id[i]
                unchanged_dict[self.params[_i]] = jnp.reshape(z1[self.remain_l[i]:self.remain_r[i]], self.par_shapes[_i])
            def single_sample_logpdf(marginals):
                dict = unchanged_dict
                for i in range(len(self.marginalize_id)):
                    _i = self.marginalize_id[i]
                    dict[self.params[_i]] = jnp.reshape(marginals[self.marginalize_l[i]:self.marginalize_r[i]],
                                                        self.par_shapes[_i])
                return -self.potential_fn(*self.model_args, **self.model_kwargs)(dict)
            logpdfs = vmap(single_sample_logpdf)(samples)
            logpdfs = jnp.asarray(logpdfs)
            x = logpdfs-qlogpdf
            log_mean_sqr = log_mean_exp(2*x)
            log_mean = log_mean_exp(x)
            #bigger = jnp.max(jnp.array([log_mean*2, log_mean_sqr]))
            #log_var = jnp.log(jnp.exp(log_mean_sqr-bigger)-jnp.exp(2*log_mean-bigger))+bigger
            return - log_mean  - cond_logdet, log_mean_sqr - log_mean*2
        return neg_log_prob
    def get_params(self):
        return self.params


