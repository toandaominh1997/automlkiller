import os
import numpy as np
import ray
from ray import tune

from autotonne.models.model_factory import ModelFactory
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from tqdm.auto import tqdm, trange
import pickle
from autotonne.utils import LOGGER


@ModelFactory.register('rl-ppo')
class PPOContainer(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = PPOrl(**kwargs)

class PPOrl(object):
    def __init__(self, env, env_config, config):
        self.config = config
        self.config['env_config'] = env_config
        self.env = env(env_config)
        self.agent = PPOTrainer(config = self.config, env = env)


    def fit(self, checkpoint = None, n_iter = 2000, save_checkpoint = 10):
        if checkpoint is None:
            checkpoint = os.path.join(os.getcwd(), 'data/checkpoint_rl.pkl')
        for idx in trange(n_iter):
            result = self.agent.train()
            LOGGER.warning('result: ', result)
            if (idx + 1) % save_checkpoint == 0:
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
        done = False
        episode_reward = 0
        obs = self.env.reset()
        actions = []
        while not done:
            action = self.agent.compute_action(obs)
            actions.append(action)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        results = {'action': actions, 'reward': episode_reward}
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

