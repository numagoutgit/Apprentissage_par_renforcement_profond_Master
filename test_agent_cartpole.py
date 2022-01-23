import gym
import torch
from DQN_cartpole import DQN
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from itertools import count
import random

class testClass:

    def __init__(self, env, model_path, eps, record = False):
        self.env = env
        self.eps = eps
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
            self.env.render()
            self.videorecorder.capture_frame()
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            if done:
                print(f'Episode finished after {t} timesteps')
                break

        self.videorecorder.close()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    test = testClass(env, 'cartpole_model', 0)
    test.test()
