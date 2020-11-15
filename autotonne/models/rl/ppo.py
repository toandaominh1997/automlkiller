import ray
from ray import tune

import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import gym
from gym.spaces import Discrete, Box
from tqdm.auto import tqdm, trange
import pickle
from autotonne.utils import LOGGER

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(
            0.0, self.end_pos, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        return [self.cur_pos], 1.0 if done else -0.1, done, {}


class PPOContainer(object):
    def __init__(self):
        super().__init__()

        self.estimator = None

class PPOrl():
    def __init__(self, env, env_config, config):
        self.config = config
        self.config['env_config'] = env_config
        self.env = env(env_config)
        self.agent = PPOTrainer(config = self.config, env = env)

        self.stop = {
            "training_iteration": 50,
            "timesteps_total": 1000000,
            "episode_reward_mean": 0.1,
        }

    def fit(self, checkpoint = None):
        for idx in trange(5):
            result = self.agent.train()
            LOGGER.warning('result: ', result)
            if (idx + 1) % 5 == 0:
                LOGGER.warning('Save checkpoint at: {}'.format(idx + 1))
                state = self.agent.save_to_object()
                with open(checkpoint, 'wb') as fp:
                    pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return result
    def predict(self, checkpoint = None):
        if checkpoint is not None:
            with open(checkpoint, 'rb') as fp:
                state = pickle.load(fp)
            self.agent.restore_from_object(state)
            # self.agen= t.restore('./checkpoint_1/checkpoint-1')
        done = False
        episode_reward = 0
        obs = self.env.reset()
        actions = []
        while not done:
            action = self.agent.compute_action(obs)
            actions.append(action)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        results = {'action': actions, 'episode_reward': episode_reward}
        return results

if __name__ =='__main__':
    config = {
        "vf_share_layers": True,
        "lr": 1e-3,  # try different lrs
        "num_workers": 8,  # parallelism
        "framework": "torch",
        "rollout_fragment_length": 10,
    }
    ray.init()
    rl = PPOrl(config = config, env = SimpleCorridor, env_config = {'corridor_length': 5})
    env = SimpleCorridor({'corridor_length': 5})
    rl.fit(checkpoint = './rllib.pkl')
    reward = rl.predict(checkpoint = './rllib.pkl')
    print('total_reward: ', reward)

