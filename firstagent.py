import random
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim

from DQN_cartpole import DQN


class Agent:
    def __init__(self, env, nb_episode, record = False):
        self.env = env
        self.nb_episode = nb_episode
        self.episode_duration = []
        self.videorecorder = VideoRecorder(env, 'videos/new_video.mp4', enabled=record)
        self.policy_net = DQN()

        self.eps = 0.99
        self.eps_decay = 0.999

    def process_state(self, state):
        '''Transform the state array into pytorch tensor'''
        return torch.from_numpy(state).unsqueeze(0).float()

    def select_action(self, state):
        sample = random.random()
        if sample < self.eps:
            self.eps *= self.eps_decay
            return torch.tensor([[random.randrange(self.env.action_space.n)]], dtype=torch.long).item()
        else:
            self.eps *= self.eps_decay
            with torch.no_grad():
                action = self.policy_net(self.process_state(state)).max(1)[1].item()
                return action

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_duration, dtype=torch.float)
        plt.title('Evaluation')
        plt.xlabel('Episodes')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def run(self):
        for i_episode in range(self.nb_episode):
            state = self.env.reset()
            for t in range(100):
                env.render()
                self.videorecorder.capture_frame()
                action = self.select_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    self.episode_duration.append(t+1)
                    self.plot_durations()
                    break
        self.videorecorder.close()
        env.close()
    

    

if __name__ == '__main__':
    plt.ion()
    env = gym.make('CartPole-v1')
    agent = Agent(env, 10, True)
    agent.run()