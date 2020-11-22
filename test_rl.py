import numpy as np
import gym
from gym.spaces import Discrete, Box

from automlkiller.reinforcementlearning import AUTORL


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

if __name__ =='__main__':

    config = {
            # "vf_share_layers": True,
            "lr": 1e-3,  # try different lrs
            "num_workers": 8,  # parallelism
            "framework": "torch",
            "rollout_fragment_length": 10,
    }

    autorl = AUTORL(
        config = config,
        env = SimpleCorridor,
        env_config = {'corridor_length': 5}
    )
    autorl.create_model(estimator = ['rl-ppo'])
    autorl.report_rl()


