import argparse
import os
import gym
import numpy as np
import pickle
from tqdm import trange
import visdom

import torch

from ddpg import DDPG
from normalized_actions import NormalizedActions
from ounoise import OrnsteinUhlenbeckActionNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from utils import save_model, vis_plot

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='discount factor for model (default: 0.01)')
parser.add_argument('--no_ou_noise', default=False, action='store_true')
parser.add_argument('--param_noise', default=False, action='store_true')
parser.add_argument('--noise_scale', type=float, default=0.2, metavar='G',
                    help='(default: 0.2)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--num_epochs', type=int, default=None, metavar='N',
                    help='number of epochs (default: None)')
parser.add_argument('--num_epochs_cycles', type=int, default=20, metavar='N')
parser.add_argument('--num_rollout_steps', type=int, default=100, metavar='N')
parser.add_argument('--num_steps', type=int, default=2000000, metavar='N',
                    help='number of training steps (default: 2000000)')
parser.add_argument('--param_noise_interval', type=int, default=50, metavar='N')
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='number of neurons in the hidden layers (default: 64)')
parser.add_argument('--number_of_train_steps', type=int, default=50, metavar='N')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--method', default='mdp', choices=['mdp', 'pr_mdp', 'nr_mdp'])
parser.add_argument('--ratio', default=10, type=int)
parser.add_argument('--flip_ratio', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='control given to adversary (default: 0.1)')
parser.add_argument('--exploration_method', default=None, choices=['mdp', 'nr_mdp', 'pr_mdp'])
parser.add_argument('--visualize', default=False, action='store_true')
args = parser.parse_args()

if args.exploration_method is None:
    args.exploration_method = args.method

args.ou_noise = not args.no_ou_noise

env = NormalizedActions(gym.make(args.env_name))
eval_env = NormalizedActions(gym.make(args.env_name))

agent = DDPG(gamma=args.gamma, tau=args.tau, hidden_size=args.hidden_size, num_inputs=env.observation_space.shape[0],
             action_space=env.action_space, train_mode=True, alpha=args.alpha, replay_size=args.replay_size)

results_dict = {'eval_rewards': [],
                'value_losses': [],
                'policy_losses': [],
                'adversary_losses': [],
                'train_rewards': []
                }
value_losses = []
policy_losses = []
adversary_losses = []

base_dir = os.getcwd() + '/models/' + args.env_name + '/'

if args.param_noise:
    base_dir += 'param_noise/'
elif args.ou_noise:
    base_dir += 'ou_noise/'
else:
    base_dir += 'no_noise/'

if args.exploration_method == args.method:
    if args.method != 'mdp':
        if args.flip_ratio:
            base_dir += 'flip_ratio_'
        base_dir += args.method + '_' + str(args.alpha) + '_' + str(args.ratio) + '/'
    else:
        base_dir += 'non_robust/'
else:
    if args.method != 'mdp':
        if args.flip_ratio:
            base_dir += 'flip_ratio_'
        base_dir += 'alternative_' + args.method + '_' + str(args.alpha) + '_' + str(args.ratio) + '/'
    else:
        base_dir += 'alternative_non_robust_' + args.exploration_method + '_' + str(args.alpha) + '_' + str(args.ratio) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number)
os.makedirs(base_dir)

ounoise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]),
                                       sigma=float(args.noise_scale) * np.ones(env.action_space.shape[0])
                                       ) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=args.noise_scale,
                                     desired_action_stddev=args.noise_scale) if args.param_noise else None


def reset_noise(a, a_noise, p_noise):
    if a_noise is not None:
        a_noise.reset()
    if p_noise is not None:
        a.perturb_actor_parameters(param_noise)


total_steps = 0
print(base_dir)

if args.num_steps is not None:
    assert args.num_epochs is None
    nb_epochs = int(args.num_steps) // (args.num_epochs_cycles * args.num_rollout_steps)
else:
    nb_epochs = 500


state = agent.Tensor([env.reset()])
eval_state = agent.Tensor([eval_env.reset()])

eval_reward = 0
episode_reward = 0
agent.train()

reset_noise(agent, ounoise, param_noise)

if args.visualize:
    vis = visdom.Visdom(env=base_dir)
else:
    vis = None

train_steps = 0
args.ratio += 1
for epoch in trange(nb_epochs):
    for cycle in range(args.num_epochs_cycles):
        with torch.no_grad():
            for t_rollout in range(args.num_rollout_steps):
                action = agent.select_action(state, ounoise, param_noise, mdp_type=args.exploration_method)
                next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

                total_steps += 1
                episode_reward += reward

                action = agent.Tensor(action)
                mask = agent.Tensor([not done])
                next_state = agent.Tensor([next_state])
                reward = agent.Tensor([reward])

                agent.store_transition(state, action, mask, next_state, reward)

                state = next_state

                if done:
                    results_dict['train_rewards'].append((total_steps, np.mean(episode_reward)))
                    episode_reward = 0
                    state = agent.Tensor([env.reset()])
                    reset_noise(agent, ounoise, param_noise)

        if len(agent.memory) > args.batch_size:
            for t_train in range(args.number_of_train_steps):
                if train_steps % args.param_noise_interval == 0 and args.param_noise:
                    # Update param_noise based on distance metric
                    episode_transitions = agent.memory.sample(args.batch_size)
                    states = torch.stack([transition[0] for transition in episode_transitions], 0)
                    unperturbed_actions = agent.select_action(states, None, None)
                    perturbed_actions = torch.stack([transition[1] for transition in episode_transitions], 0)

                    ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(),
                                                     unperturbed_actions.cpu().numpy())
                    param_noise.adapt(ddpg_dist)

                adversary_update = (train_steps % args.ratio == 0 and not args.flip_ratio) or (train_steps % args.ratio != 0 and args.flip_ratio)

                value_loss, policy_loss, adversary_loss = agent.update_parameters(batch_size=args.batch_size,
                                                                                  mdp_type=args.method,
                                                                                  adversary_update=adversary_update,
                                                                                  exploration_method=args.exploration_method)
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                adversary_losses.append(adversary_loss)
                train_steps += 1

            results_dict['value_losses'].append((total_steps, np.mean(value_losses)))
            results_dict['policy_losses'].append((total_steps, np.mean(policy_losses)))
            results_dict['adversary_losses'].append((total_steps, np.mean(adversary_losses)))
            del value_losses[:]
            del policy_losses[:]
            del adversary_losses[:]
        with torch.no_grad():
            for t_rollout in range(args.num_rollout_steps):
                action = agent.select_action(eval_state, mdp_type='mdp')

                next_eval_state, reward, done, _ = eval_env.step(action.cpu().numpy()[0])
                eval_reward += reward

                next_eval_state = agent.Tensor([next_eval_state])

                eval_state = next_eval_state
                if done:
                    results_dict['eval_rewards'].append((total_steps, eval_reward))
                    eval_state = agent.Tensor([eval_env.reset()])
                    eval_reward = 0
        save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms,
                   rew_rms=agent.ret_rms)
        with open(base_dir + '/results', 'wb') as f:
            pickle.dump(results_dict, f)

        vis_plot(vis, results_dict)

with open(base_dir + '/results', 'wb') as f:
    pickle.dump(results_dict, f)
save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir, obs_rms=agent.obs_rms, rew_rms=agent.ret_rms)

env.close()
