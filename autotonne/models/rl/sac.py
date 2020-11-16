import os
import numpy as np
import ray
from ray import tune

from autotonne.models.model_factory import ModelFactory
from ray.rllib.agents.sac import SACTrainer
from ray.tune.logger import pretty_print
from tqdm.auto import tqdm, trange
import pickle
from autotonne.utils import LOGGER


@ModelFactory.register('rl-sac')
class SACContainer(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = SACrl(**kwargs)

class SACrl(object):
    def __init__(self, env, env_config, config):
        self.config = config
        self.config['env_config'] = env_config
        self.env = env(env_config)
        self.agent = SACTrainer(config = self.config, env = env)


    def fit(self, checkpoint = None):
        if checkpoint is None:
            checkpoint = os.path.join(os.getcwd(), 'data/checkpoint_rl.pkl')
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
