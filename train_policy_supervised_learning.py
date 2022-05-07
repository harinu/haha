from __future__ import division
import numpy as np
from itertools import count
import os, sys
#int("sys:",sys.argv)
#print("length:" ,len(sys.argv))
from networks import PolicyNN
from utils import *
from environment import KGEnvironment
from BFS.KB import KB
from BFS.BFS import BFS
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyperparameters
state_dim = 200
action_space = 400
eps_start = 1
eps_end = 0.1
epe_decay = 1000
replay_memory_size = 10000
batch_size = 128
embedding_dim = 100
gamma = 0.99
target_update_freq = 1000
max_steps = 50
max_steps_test = 50

#print("sysargv",sys.argv)
relation = sys.argv[1]
#print("sysargv1",sys.argv[1])

dataPath = '../NELL-995/'
#print("sysargv2",sys.argv[2])

model_dir = '../models'
model_name = 'policy_supervised_' + relation
# episodes = int(sys.argv[2])
graphpath = dataPath + 'tasks/' + relation + '/' + 'graph.txt'
relationPath = dataPath + 'tasks/' + relation + '/' + 'train_pos'


class PolicyNetwork(nn.Module):

    # TODO: Add regularization to policy neural net and add regularization losses to total loss
    def __init__(self, state_dim, action_space, learning_rate=0.0001):
        super(PolicyNetwork, self).__init__()
        self.action_space = action_space
        self.policy_nn = PolicyNN(state_dim, action_space)
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=learning_rate)

    def forward(self, state):
        action_prob = self.policy_nn(state)
        return action_prob

    def compute_loss(self, action_prob, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob))
        return loss
    
    def compute_loss_rl(self, action_prob, target, action):
        # TODO: Add regularization loss
        action_mask = F.one_hot(action, num_classes=self.action_space) > 0
        picked_action_prob = action_prob[action_mask]
        loss = torch.sum(-torch.log(picked_action_prob)*target)
        return loss

def train_deep_path():

    policy_network = PolicyNetwork(state_dim, action_space).to(device)
    f = open(relationPath)
    train_data = f.readlines()
    f.close()
    num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500
    else:
        num_episodes = num_samples

    for episode in range(num_samples):
        print("Episode %d" % episode)
        print('Training Sample:', train_data[episode % num_samples][:-1])

        env = KGEnvironment(dataPath, train_data[episode % num_samples])
        sample = train_data[episode % num_samples].split()
        # good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        try:
            good_episodes = teacher(sample[0], sample[1], 5, env, graphpath)
        except Exception as e:
            print('Cannot find a path')
            continue

        for item in good_episodes:
            state_batch = []
            action_batch = []
            for t, transition in enumerate(item):
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = torch.FloatTensor(state_batch).squeeze(dim=1).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            prediction = policy_network(state_batch)
            loss = policy_network.compute_loss(prediction, action_batch)
            loss.backward()

            policy_network.optimizer.step()

    # save model
    print("Saving model to disk...")
    torch.save(policy_network.cpu(), os.path.join(model_dir, model_name + '.pt'))


def test(test_episodes):

    f = open(relationPath)
    test_data = f.readlines()
    f.close()
    test_num = len(test_data)

    test_data = test_data[-test_episodes:]
    print(len(test_data))
    success = 0

    policy_network = torch.load(os.path.join(model_dir, model_name + '.pt')).to(device)
    print('Model reloaded')
    for episode in range(len(test_data)):
        print('Test sample %d: %s' % (episode, test_data[episode][:-1]))
        env = KGEnvironment(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
        for t in count():
            state_vec = torch.from_numpy(env.idx_state(state_idx)).float().to(device)
            action_probs = policy_network(state_vec)
            action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs.detach().numpy()))
            reward, new_state, done = env.interact(state_idx, action_chosen)
            if done or t == max_steps_test:
                if done:
                    print('Success')
                    success += 1
                print('Episode ends\n')
                break
            state_idx = new_state

    print('Success percentage:', success / test_episodes)

if __name__ == "__main__":
    train_deep_path()

