from tqdm import tqdm
import jax.numpy as jnp
from jax import grad,random,jit,device_get
import numpyro.distributions as dist
import numpy as np
from numpyro.infer.hmc_util import dual_averaging, welford_covariance, build_adaptation_schedule
import math
from numpyro.diagnostics import print_summary, effective_sample_size
import click
from algorithm import PMHMC, VIP, PMHMCH
import importlib
import time
from numpyro.handlers import seed
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpyro

def logmeanexp(x):
    m = jnp.max(x)
    return m + jnp.log(jnp.mean(jnp.exp(x-m)))


def leapfrog(dV_dq, compiled_step, theta, z, theta_rho, z_rho, eps, M, theta_m, z_m):
    theta_dir, z_dir = dV_dq(theta, z)
    theta_rho -= 0.5 * eps * theta_dir
    z_rho -= 0.5 * eps * z_dir
    for _ in range(M - 1):
        theta, z, theta_rho, z_rho = compiled_step(eps, theta_m, z_m, theta, z, theta_rho, z_rho)

    theta += eps * theta_rho * theta_m
    z += eps * z_rho * z_m
    theta_dir, z_dir = dV_dq(theta, z)
    theta_rho -= 0.5 * eps * theta_dir
    z_rho -= 0.5 * eps * z_dir
    return theta, z, theta_rho, z_rho


def splitting(dV_dq, compiled_step, theta, z, theta_rho, z_rho, eps, M, theta_m, z_m):
    z_sqrt_m = jnp.sqrt(z_m)
    discounted_eps = eps * z_sqrt_m  # for operator splitting
    sin_discounted = jnp.sin(discounted_eps)
    cos_discounted = jnp.cos(discounted_eps)
    sin_discounted_mul_z_sqrt_m = sin_discounted*z_sqrt_m
    sin_discounted_div_z_sqrt_m = sin_discounted / z_sqrt_m
    #print('  ',eps, discounted_eps)
    theta_dir, z_dir = dV_dq(theta, z)
    theta_rho -= 0.5 * eps * theta_dir
    z_rho -= 0.5 * eps * z_dir

    for _ in range(M - 1):
        theta, z, theta_rho, z_rho = compiled_step(eps, theta_m, sin_discounted_mul_z_sqrt_m, cos_discounted, sin_discounted_div_z_sqrt_m, theta, z, theta_rho, z_rho)
    theta += eps * theta_rho * theta_m
    z_temp = z

    z = sin_discounted_mul_z_sqrt_m * z_rho + cos_discounted * z
    z_rho = cos_discounted * z_rho - sin_discounted_div_z_sqrt_m * z_temp
    theta_dir, z_dir = dV_dq(theta, z)
    theta_rho -= 0.5 * eps * theta_dir
    z_rho -= 0.5 * eps * z_dir
    #print(theta,theta_rho)
    return theta, z, theta_rho, z_rho

class HMC:
    def __init__(self, neg_log_prob, iterate, recompile, rng_key, integrator, theta_dims, x_dims, N, lowest_epsilon=0.001):
        self.neg_log_prob = jit(neg_log_prob)
        self.iterate = iterate
        self.recompile = recompile
        self.theta_dims = theta_dims
        self.theta = jnp.zeros(theta_dims)
        self.x_dims = x_dims
        self.N = N
        self.z = jnp.zeros(self.N * self.x_dims)
        self.key = rng_key
        if integrator == 'leapfrog':
            self.integrator = leapfrog
        else:
            self.integrator = splitting
        self.lowest_epsilon = lowest_epsilon
        self.init_functions()
        self.gradient_evaluation = 0
    def init_functions(self):
        def log_estimator(theta,z):
            v = jnp.concatenate([theta,z])
            neg_log_prob = self.neg_log_prob(v)
            return -neg_log_prob
        def log_density_z(z):
            return jnp.sum(dist.Normal(0,1).log_prob(z))
        log_estimator_grad = grad(log_estimator,argnums=(0,1))
        log_density_z_grad = grad(log_density_z)
        def dV_dq(theta,z):
            theta_grad, z_grad = log_estimator_grad(theta,z)
            if self.integrator == leapfrog:
                z_grad += log_density_z_grad(z)
            return -theta_grad, -z_grad
        self.dV_dq = jit(dV_dq)
        self.log_estimator = log_estimator
        self.log_density_z = log_density_z
        if self.integrator == leapfrog:
            def splitting_step(eps, theta_m, z_m, theta, z, theta_rho, z_rho):
                theta += eps * theta_rho * theta_m
                z += eps * z_rho * z_m
                theta_dir, z_dir = dV_dq(theta, z)
                theta_rho -= eps * theta_dir
                z_rho -= eps * z_dir
                return theta, z, theta_rho, z_rho
        else:
            def splitting_step(eps, theta_m, sin_discounted_mul_z_sqrt_m, cos_discounted,
                               sin_discounted_div_z_sqrt_m, theta, z, theta_rho, z_rho):
                theta += eps * theta_rho * theta_m
                z_temp = z
                z = sin_discounted_mul_z_sqrt_m * z_rho + cos_discounted * z
                z_rho = cos_discounted * z_rho - sin_discounted_div_z_sqrt_m * z_temp
                theta_dir, z_dir = dV_dq(theta, z)
                theta_rho -= eps * theta_dir
                z_rho -= eps * z_dir
                return theta, z, theta_rho, z_rho
        self.compiled_step = jit(splitting_step)


    def get_init_parameters(self):
        return self.theta, self.z

    def step(self, eps, trajectory_length, theta, z, mass_matrix_sqrt_inv, inverse_mass_matrix, sampling = False):
        finfo = jnp.finfo(jnp.result_type(eps))
        eps = jnp.clip(eps, a_min=self.lowest_epsilon, a_max=finfo.max)
        M = jnp.ceil(trajectory_length / eps).astype(int)
        if sampling:
            self.gradient_evaluation += M + 1
        eps = trajectory_length / M
        # p <- p - \eps/2 * dV/dq
        # q <- q + \eps * p
        # p <- p - \eps/2 * dV/dq


        theta_key, z_key, self.key = random.split(self.key,3)
        theta_sqrt_m = mass_matrix_sqrt_inv[:self.theta_dims]
        z_sqrt_m = mass_matrix_sqrt_inv[self.theta_dims:]
        theta_rho = random.normal(theta_key,theta.shape) / theta_sqrt_m
        z_rho = random.normal(z_key,z.shape) / z_sqrt_m
        #print(theta, z, theta_rho,z_rho)
        theta_m = inverse_mass_matrix[:self.theta_dims]
        z_m = inverse_mass_matrix[self.theta_dims:]

        theta_last = theta
        theta_rho_last = theta_rho
        z_last = z
        z_rho_last = z_rho

        theta, z, theta_rho, z_rho = self.integrator(self.dV_dq, self.compiled_step, theta, z, theta_rho, z_rho, eps, M, theta_m, z_m)

        log_density_last = self.log_estimator(theta_last,z_last) + self.log_density_z(z_last) - jnp.sum(jnp.square(theta_rho_last) * theta_m)/2 - jnp.sum(jnp.square(z_rho_last) * z_m)/2
        log_density = self.log_estimator(theta, z) + self.log_density_z(z) - jnp.sum(jnp.square(theta_rho) * theta_m)/2 - jnp.sum(jnp.square(z_rho) * z_m)/2
        log_ratio = 0 if log_density - log_density_last > 0 else log_density - log_density_last
        #print(log_density, log_density_last)
        ratio = jnp.exp(log_ratio)
        if jnp.isnan(ratio):
            ratio = 0.
        self.key, sample_key = random.split(self.key)
        rand_val = random.uniform(sample_key)

        return theta,z,rand_val > ratio, ratio

    @property
    def get_dimension(self):
        return int(self.N*self.x_dims + self.theta_dims)

@click.command()
@click.option('--warmup_steps', default=10000)
@click.option('--sample_steps', default=100000)
@click.option('--rng_key', default=0)
@click.option('--integrator', default='splitting')
@click.option('--j', default=8)
@click.option('--rho', default=1.)
@click.option('--trajectory_length', default=math.pi)
@click.option('--init_epsilon', default=0.1)
@click.option('--lowest_epsilon', default=0.001)
@click.option('--target_accept_prob', default=0.8)
@click.option('--samples', default=8, help = 'Number of samples in PMHMC.')
@click.option('--approximator', default = 'VariationalInference', help = 'Approximator in PMHMC.')
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference.')
@click.option('--training_steps', default=1000000)
@click.option('--training_lr', default=1e-3)
@click.option('--algorithm', default='PMHMC')
@click.option('--lamb', default=10.)

def main(warmup_steps,sample_steps,rng_key,integrator,j, rho,trajectory_length,init_epsilon, lowest_epsilon,target_accept_prob,
         samples, approximator, model, training_steps, training_lr, algorithm, lamb):
    model_name = model
    module = importlib.import_module('model')
    model = getattr(module, model_name)()
    if model_name == 'EightSchools':
        model.set_parameters(j,rho)
    model.set_lambda(lamb)
    model_path = 'result2/{}'.format(model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    alg_path = os.path.join(model_path,'PMHMC_{}'.format(approximator))
    if not os.path.exists(alg_path):
        os.mkdir(alg_path)
    result_path = os.path.join(alg_path,'{samples}_{warm_up_steps}_{sample_steps}_{rng_key}'.format(
        samples = samples,
        warm_up_steps=warmup_steps,
        sample_steps=sample_steps,
        rng_key=rng_key
    ))
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_file = os.path.join(result_path,'result')
    #jax.profiler.start_trace(os.path.join(result_path,'tensorboard'))

    hmc_key, algo_key, postprocess_key = random.split(random.PRNGKey(rng_key),3)
    algo_module = importlib.import_module('algorithm')
    algo = getattr(algo_module, algorithm)
    algor = algo(model.model, algo_key, *model.args(), num_sample=samples, approximate_strategy=approximator,
                vi_steps=training_steps, vi_step_size=training_lr, model_class = model, **model.kwargs())
    theta_dims, x_dims, neg_log_prob, postprocess, iterate, recompile = algor.compile(model.parameters(),num_warmup=warmup_steps, num_samples=sample_steps)
    hmc = HMC(neg_log_prob, iterate, recompile, hmc_key,integrator, theta_dims, x_dims, samples, lowest_epsilon)
    ss_init, ss_update = dual_averaging()
    mm_init, mm_update, mm_final = welford_covariance(diagonal=True)
    ss_state = ss_init(jnp.log(10 * init_epsilon))
    mm_state = mm_init(hmc.get_dimension)
    adaptation_schedule = jnp.array(build_adaptation_schedule(warmup_steps))
    num_windows = len(adaptation_schedule)
    inverse_mass_matrix = jnp.ones(hmc.get_dimension)
    mass_matrix_sqrt = mass_matrix_sqrt_inv = inverse_mass_matrix
    theta,z = hmc.get_init_parameters()
    warmup_accept = 0
    warmup_step = 0
    for window_id in tqdm(range(num_windows)):
        window = adaptation_schedule[window_id]
        warmup_bar = tqdm(range(window[0],window[1]+1))
        for all in warmup_bar:
            warmup_step += 1
            log_step_size, _, *_ = ss_state
            step_size = jnp.exp(log_step_size)
            theta, z, rj, ratio = hmc.step(step_size, trajectory_length, theta, z, mass_matrix_sqrt_inv, inverse_mass_matrix)
            if not rj:
                warmup_accept += 1
            accepted = warmup_accept/warmup_step
            ss_state = ss_update(target_accept_prob - ratio, ss_state)
            #print(jnp.exp(log_step_size), accepted,jnp.concatenate([theta,z]))
            if window_id < num_windows - 1:
                mm_state = mm_update(jnp.concatenate([theta,z]),mm_state)
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
    zs = []
    rejected = 0
    bar = tqdm(range(1,sample_steps + 1))
    for all in bar:
        theta, z, rj, ratio = hmc.step(eps,trajectory_length,theta,z, mass_matrix_sqrt_inv, inverse_mass_matrix, sampling = True)
        if rj:
            rejected += 1
        res.append(theta.copy())
        zs.append(z.copy())
        if all % 100 == 0:
            bar.set_description("reject rate: {}".format(rejected / all))
    print("reject rate: {}; mean: {}; std: {}".format(rejected / sample_steps,jnp.mean(jnp.asarray(res)),jnp.std(jnp.asarray(res))))
    res = np.asarray(res)
    zs = np.array(zs)
    #res = np.expand_dims(res,axis=0)
    res = postprocess(res,zs, postprocess_key)
    end_time = time.time()
    overall_time = end_time - start_time
    #print(res)
    print_summary(res)
    esss = []
    essps = []
    esspgrad = []
    hmc.gradient_evaluation*=samples
    for par in res.keys():
        if not par.endswith("_base"):
            ess = effective_sample_size(device_get(res[par]))
            esss.extend(ess)
            essps.extend(ess/overall_time)
            esspgrad.extend(ess/hmc.gradient_evaluation)
            print(par, np.mean(ess/overall_time), np.mean(ess/hmc.gradient_evaluation))
    print("%.f"%np.min(esss),"%.f"%np.mean(esss),"%.f"%np.max(esss),sep=' ')
    print("%.f"%np.min(essps),"%.f"%np.mean(essps),"%.f"%np.max(essps),sep=' ')
    print("%.2f"%np.min(esspgrad),"%.2f"%np.mean(esspgrad),"%.2f"%np.max(esspgrad),sep=' ')

    with open(result_file,"w") as f:
        print(eps,file=f)
        print("overall time:", overall_time, file=f)
        print("%.f" % np.min(esss), "%.f" % np.mean(esss), "%.f" % np.max(esss), sep=' ', file=f)
        print("%.f" % np.min(essps), "%.f" % np.mean(essps), "%.f" % np.max(essps), sep=' ', file=f)
        print("%.2f" % np.min(esspgrad), "%.2f" % np.mean(esspgrad), "%.2f" % np.max(esspgrad), sep=' ', file=f)


if __name__ == '__main__':
    main()