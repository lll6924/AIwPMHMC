from tqdm import tqdm
import jax.numpy as jnp
from jax import grad,random,jit,device_get
import numpy as np
from numpyro.infer.hmc_util import dual_averaging, welford_covariance, build_adaptation_schedule
import math
from numpyro.diagnostics import print_summary, effective_sample_size
import click
import importlib
import time
import os

def logmeanexp(x):
    m = jnp.max(x)
    return m + jnp.log(jnp.mean(jnp.exp(x-m)))

def leapfrog(dV_dq, z, z_rho, eps, M, z_m):
    z_dir = dV_dq(z)
    z_rho -= 0.5 * eps * z_dir
    for _ in range(M - 1):
        z += eps * z_rho * z_m
        z_dir = dV_dq(z)
        z_rho -= eps * z_dir
    z += eps * z_rho * z_m
    z_dir = dV_dq(z)
    z_rho -= 0.5 * eps * z_dir
    return z, z_rho


class HMC:
    def __init__(self, neg_log_prob, rng_key, z_dims, lowest_epsilon=0.001):
        self.neg_log_prob = jit(neg_log_prob)
        self.z_dims = z_dims
        self.key = rng_key
        self.integrator = leapfrog
        self.lowest_epsilon = lowest_epsilon
        self.init_functions()
        self.z = jnp.zeros(self.z_dims)
        self.gradient_evaluation = 0

    def get_init_parameters(self):
        return self.z
    def init_functions(self):
        def log_estimator(z):
            neg_log_prob = self.neg_log_prob(z)
            return -neg_log_prob
        log_estimator_grad = grad(log_estimator,argnums=(0))
        def dV_dq(z):
            z_grad = log_estimator_grad(z)
            return -z_grad
        self.dV_dq = jit(dV_dq)
        self.log_estimator = log_estimator

    def step(self, eps, trajectory_length, z, mass_matrix_sqrt_inv, inverse_mass_matrix, sampling = False):
        finfo = jnp.finfo(jnp.result_type(eps))
        eps = jnp.clip(eps, a_min=self.lowest_epsilon, a_max=finfo.max)
        M = jnp.ceil(trajectory_length / eps).astype(int)
        if sampling:
            self.gradient_evaluation += M+1
        eps = trajectory_length / M
        # p <- p - \eps/2 * dV/dq
        # q <- q + \eps * p
        # p <- p - \eps/2 * dV/dq


        theta_key, z_key, self.key = random.split(self.key,3)
        z_sqrt_m = mass_matrix_sqrt_inv
        z_rho = random.normal(z_key,z.shape) / z_sqrt_m
        z_m = inverse_mass_matrix

        z_last = z
        z_rho_last = z_rho

        z, z_rho = self.integrator(self.dV_dq, z, z_rho, eps, M, z_m)

        log_density_last = self.log_estimator(z_last) - jnp.sum(jnp.square(z_rho_last) * z_m)/2
        log_density = self.log_estimator(z) - jnp.sum(jnp.square(z_rho) * z_m)/2
        log_ratio = 0 if log_density - log_density_last > 0 else log_density - log_density_last
        #print(log_density, log_density_last)
        ratio = jnp.exp(log_ratio)
        if jnp.isnan(ratio):
            ratio = 0.
        self.key, sample_key = random.split(self.key)
        rand_val = random.uniform(sample_key)

        if rand_val > ratio:
            z = z_last

        return z,rand_val > ratio, ratio

    @property
    def get_dimension(self):
        return int(self.z_dims)

@click.command()
@click.option('--warmup_steps', default=10000)
@click.option('--sample_steps', default=100000)
@click.option('--rng_key', default=0)
@click.option('--j', default=8)
@click.option('--rho', default=1.)
@click.option('--trajectory_length', default=math.pi)
@click.option('--init_epsilon', default=0.1)
@click.option('--lowest_epsilon', default=0.001)
@click.option('--target_accept_prob', default=0.8)
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference.')
@click.option('--training_steps', default=1000000)
@click.option('--training_lr', default=1e-3)
@click.option('--algorithm', default='HMC')
@click.option('--lamb', default=10.)
@click.option('--samples', default=1, help = 'Number of samples in PMHMC.')
@click.option('--approximator', default = 'VariationalInference', help = 'Approximator in PMHMC.')
def main(warmup_steps,sample_steps,rng_key,j, rho,trajectory_length,init_epsilon, lowest_epsilon,target_accept_prob,
        model, training_steps, training_lr, algorithm, lamb, samples, approximator):
    model_name = model
    module = importlib.import_module('model')
    model = getattr(module, model_name)()
    if model_name.startswith('EightSchools'):
        model.set_parameters(j,rho)
    model.set_lambda(lamb)
    model_path = 'result2/{}_{}_{}'.format(model_name,rho,lamb)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    alg_path = os.path.join(model_path,'{}_{}'.format(algorithm,samples))
    if not os.path.exists(alg_path):
        os.mkdir(alg_path)
    result_path = os.path.join(alg_path,'{warm_up_steps}_{sample_steps}_{rng_key}'.format(
        warm_up_steps=warmup_steps,
        sample_steps=sample_steps,
        rng_key=rng_key
    ))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_file = os.path.join(result_path,'result')
    #jax.profiler.start_trace(os.path.join(result_path,'tensorboard'))

    hmc_key, algo_key, postprocess_key = random.split(random.PRNGKey(rng_key), 3)
    algo_module = importlib.import_module('algorithm')
    algo = getattr(algo_module, algorithm)
    print(algorithm)
    if algorithm!='PMHMC2' and algorithm!='PMHMCH2':
        algor = algo(model.model, algo_key, *model.args(),
                     vi_steps=training_steps, vi_step_size=training_lr, model_class = model, **model.kwargs())
        x_dims, neg_log_prob, postprocess = algor.compile(model.parameters(),num_warmup=warmup_steps, num_samples=sample_steps)
    else:
        algor = algo(model.model, algo_key, *model.args(), num_sample=samples,
                     approximate_strategy=approximator,
                     vi_steps=training_steps, vi_step_size=training_lr, model_class=model, **model.kwargs())
        n_remained, n_marginalized, neg_log_prob, postprocess, _,_  = algor.compile(model.parameters(),num_warmup=warmup_steps, num_samples=sample_steps)
        x_dims = n_remained + n_marginalized * samples
    hmc = HMC(neg_log_prob, hmc_key, x_dims, lowest_epsilon)
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=True)
    ss_state = ss_init(jnp.log(10 * init_epsilon))
    mm_state = mm_init(hmc.get_dimension)
    adaptation_schedule = jnp.array(build_adaptation_schedule(warmup_steps))
    num_windows = len(adaptation_schedule)
    inverse_mass_matrix = jnp.ones(hmc.get_dimension)
    mass_matrix_sqrt = mass_matrix_sqrt_inv = inverse_mass_matrix
    z = hmc.get_init_parameters()
    warmup_accept = 0
    warmup_step = 0
    for window_id in tqdm(range(num_windows)):
        window = adaptation_schedule[window_id]
        warmup_bar = tqdm(range(window[0],window[1]+1))
        for all in warmup_bar:
            warmup_step += 1
            log_step_size, _, *_ = ss_state
            step_size = jnp.exp(log_step_size)
            z, rj, ratio = hmc.step(step_size, trajectory_length, z, mass_matrix_sqrt_inv, inverse_mass_matrix)
            if not rj:
                warmup_accept += 1
            accepted = warmup_accept/warmup_step
            ss_state = ss_update(target_accept_prob - ratio, ss_state)
            #print(jnp.exp(log_step_size), accepted,jnp.concatenate([theta,z]))
            if window_id < num_windows - 1:
                mm_state = mm_update(z,mm_state)
            if all % 100 == 0:
                warmup_bar.set_description("reject rate: {}  step size: {}".format(1-accepted,step_size))

        if window_id < num_windows - 1:
            inverse_mass_matrix, mass_matrix_sqrt, mass_matrix_sqrt_inv = mm_final(
                mm_state, regularize=True
            )
            mm_state = mm_init(hmc.get_dimension)
            log_step_size, _, *_ = ss_state
            ss_state = ss_init(jnp.log(10) + log_step_size)
        #print(inverse_mass_matrix, mass_matrix_sqrt_inv)

    _, log_step_size_avg, *_ = ss_state
    eps = jnp.exp(log_step_size_avg)
    print('step size after warmup: ', eps)

    #print('inverse mass matrix: ', inverse_mass_matrix)

    start_time = time.time()

    res = []
    rejected = 0
    bar = tqdm(range(1,sample_steps + 1))
    for all in bar:
        z, rj, ratio = hmc.step(eps,trajectory_length,z, mass_matrix_sqrt_inv, inverse_mass_matrix, sampling = True)
        if rj:
            rejected += 1
        res.append(z.copy())
        if all % 100 == 0:
            bar.set_description("reject rate: {}".format(rejected / all))
    print("reject rate: {}; mean: {}; std: {}".format(rejected / sample_steps,jnp.mean(jnp.asarray(res)),jnp.std(jnp.asarray(res))))
    res = np.asarray(res)
    print(res.shape)
    if algorithm == 'PMHMC2' or algorithm=='PMHMCH2':
        thetas = res[...,:n_remained]
        zs = res[...,n_remained:]
        print(thetas.shape, zs.shape, res.shape)
        res = postprocess(thetas, zs, postprocess_key)
    else:
        res = np.expand_dims(res, axis=0)

        res = postprocess(res)
    end_time = time.time()
    overall_time = end_time - start_time
    #print(res)
    print_summary(res)
    esss = []
    essps = []
    esspgrad = []
    if type(res) == dict:
        for par in res.keys():
            if not par.endswith("_base"):
                ess = effective_sample_size(device_get(res[par]))
                #print("ESS for ",par, ": ", ess)
                #print("ESS/s for ",par, ": ", ess/overall_time)
                #print("ESS/grad for ",par, ": ", ess/hmc.gradient_evaluation)
                if type(ess) == np.float64 :
                    esss.append(ess)
                    essps.append(ess / overall_time)
                    esspgrad.append(ess / hmc.gradient_evaluation)
                else:
                    esss.extend(ess)
                    essps.extend(ess/overall_time)
                    esspgrad.extend(ess/hmc.gradient_evaluation)
                print(par, np.mean(ess / overall_time), np.mean(ess / hmc.gradient_evaluation))
    else:
        ess = effective_sample_size(device_get(res))
        # print("ESS for ",par, ": ", ess)
        # print("ESS/s for ",par, ": ", ess/overall_time)
        # print("ESS/grad for ",par, ": ", ess/hmc.gradient_evaluation)
        esss.extend(ess)
        essps.extend(ess / overall_time)
        esspgrad.extend(ess / hmc.gradient_evaluation)
    print("%.f"%np.min(esss),"%.f"%np.mean(esss),"%.f"%np.max(esss),sep=' ')
    print("%.f"%np.min(essps),"%.f"%np.mean(essps),"%.f"%np.max(essps),sep=' ')
    print("%.2f"%np.min(esspgrad),"%.2f"%np.mean(esspgrad),"%.2f"%np.max(esspgrad),sep=' ')

    with open(result_file,"w") as f:
        print(eps,file=f)
        print("overall time:", overall_time, file=f)
        print("%.f" % np.min(esss), "%.f" % np.mean(esss), "%.f" % np.max(esss), sep=' ', file=f)
        print("%.f" % np.min(essps), "%.f" % np.mean(essps), "%.f" % np.max(essps), sep=' ', file=f)
        print("%.2f" % np.min(esspgrad), "%.2f" % np.mean(esspgrad), "%.2f" % np.max(esspgrad), sep=' ', file=f)
    #print(effective_sample_size(res))


if __name__ == '__main__':
    main()