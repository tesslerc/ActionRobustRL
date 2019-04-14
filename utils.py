import os
import torch
import numpy as np


def save_model(actor, adversary, obs_rms, rew_rms, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)
    vars_path = "{}/ddpg_vars".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    torch.save(actor.state_dict(), actor_path)
    torch.save(adversary.state_dict(), adversary_path)

    var_dict = {'obs_rms_mean': None, 'obs_rms_var': None, 'rew_rms_mean': None, 'rew_rms_var': None}
    if obs_rms is not None:
        var_dict['obs_rms_mean'] = obs_rms.mean
        var_dict['obs_rms_var'] = obs_rms.var
    if rew_rms is not None:
        var_dict['rew_rms_mean'] = rew_rms.mean
        var_dict['rew_rms_var'] = rew_rms.var
    torch.save(var_dict, vars_path)


def load_model(agent, basedir=None):
    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)
    vars_path = "{}/ddpg_vars".format(basedir)

    print('Loading models from {} {}'.format(actor_path, adversary_path))
    agent.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    agent.adversary.load_state_dict(torch.load(adversary_path, map_location=lambda storage, loc: storage))

    var_dict = torch.load(vars_path)
    if var_dict['obs_rms_mean'] is not None:
        agent.obs_rms.mean = var_dict['obs_rms_mean']
        agent.obs_rms.var = var_dict['obs_rms_var']
        agent.normalize_observations = True
    else:
        agent.obs_rms = None
        agent.normalize_observations = False


# Borrowed from openai baselines running_mean_std.py
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def vis_plot(viz, log_dict):
    if viz is not None:
        for field in log_dict:
            if len(log_dict[field]) > 0:
                _, values = zip(*log_dict[field])

                plot_data = np.array(log_dict[field])
                viz.line(X=plot_data[:, 0], Y=plot_data[:, 1], win=field,
                         opts=dict(title=field, legend=[field]))
