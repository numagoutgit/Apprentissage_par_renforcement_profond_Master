import random
import gym
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim

from DQN_cartpole import DQN
from ReplayMemory import Transition, ReplayMemory


class Agent:
    def __init__(self, env, nb_episode):
        self.env = env
        self.nb_episode = nb_episode
        self.episode_duration = []
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = 30
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.batch_size = 128

        self.gamma = 0.90
        self.eps = 0.99
        self.eps_decay = 0.999

    def process_state(self, state):
        '''Transform the state array into pytorch tensor'''
        if state is None:
            return None
        return torch.from_numpy(state).unsqueeze(0).float()

    def select_action(self, state):
        sample = random.random()
        if sample < self.eps:
            self.eps *= self.eps_decay
            return torch.tensor([[random.randrange(self.env.action_space.n)]], dtype=torch.long)
        else:
            self.eps *= self.eps_decay
            with torch.no_grad():
                action = self.policy_net(self.process_state(state)).max(1)[1].view(1,1)
                return action

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_duration, dtype=torch.float)
        plt.title('Training ...')
        plt.xlabel('Episodes')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_indice = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Calcul de Q(s,a) pour l'etat courant
        state_action_values = self.policy_net(state_batch).gather(1,action_batch).squeeze()

        # Calcul de Q(s',a) pour l'etat suivant
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_indice] = self.target_net(non_final_next_states).max(1)[0]

        # Calcul cible
        cible = ((next_state_values * self.gamma) + reward_batch)

        # Calcul coût
        lossFunction = nn.MSELoss()
        loss = lossFunction(state_action_values, cible)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        for i_episode in range(self.nb_episode):
            state = self.env.reset()
            for t in count():
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                if done:
                    next_state = None

                self.memory.push(self.process_state(state), action, self.process_state(next_state), reward)
                state = next_state
                self.optimize_model()
                if done:
                    self.episode_duration.append(t+1)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict((self.policy_net.state_dict()))
                self.policy_net.save_model()
        self.policy_net.save_model()
        plt.show()

    

if __name__ == '__main__':
    plt.ion()
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(env, 2000)
    agent.run()