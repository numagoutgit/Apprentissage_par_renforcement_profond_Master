import gym
import torch
from DQN_cartpole import DQN
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from itertools import count
import random
import sys

class testClass:

    def __init__(self, env, model_path, eps, record = False, render = True):
        self.env = env
        self.eps = eps
        self.render = render
        self.videorecorder = VideoRecorder(env, 'videos/new_video.mp4', enabled=record)
        self.model = DQN()
        self.model.load_state_dict(torch.load(model_path))

    def process_state(self, state):
        if state is None:
            return None
        return torch.from_numpy(state).unsqueeze(0).float()

    def select_action(self, state):
        sample = random.random()
        if sample < self.eps:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.model(self.process_state(state)).max(1)[1].view(1,1)
                return action

    def test(self):
        state = self.env.reset()
        print('Testing model ..')
        for t in count():
            if self.render:
                self.env.render()
                self.videorecorder.capture_frame()
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            if done:
                print(f'Episode finished after {t+1} timesteps')
                break

            if (t+1)%1000 == 0:
                print(f'{t+1} timesteps')

        if self.render:
            self.videorecorder.close()

if __name__ == '__main__':
    path = sys.argv[1]
    epsilon = int(sys.argv[2])
    record = sys.argv[3] == 'True'
    wrapped = (sys.argv[4]) == 'True'
    render = (sys.argv[5]) == 'True'
    if wrapped:
        env = gym.make('CartPole-v1')
    else:
        env = gym.make('CartPole-v1').unwrapped
    test = testClass(env, path, epsilon, record, render)
    test.test()
