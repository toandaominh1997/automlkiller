
import ray

from autotonne.models.model_factory import ModelFactory
from autotonne.utils import LOGGER


class AUTORL(object):
    def __init__(self,
                 config,
                 env,
                 env_config,
                 ):
        ray.init()
        self.config = config
        self.env = env
        self.env_config = env_config
        self.estimator = {}
        self.model = {}
        self.metrics = {}
        self.estimator_params = {}

    def create_model(self,
                     estimator,
                     n_jobs  = -1,
                     estimator_params = {}
                     ):

        estimator_model = {}
        if estimator is None:
            if len(self.estimator.keys()) > 0:
                for name_model, estimator in self.estimator.items():
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model,
                                                                                   env = self.env,
                                                                                   env_config = self.env_config,
                                                                                   config = self.config,
                                                                                   **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = estimator

            else:
                for name_model in ModelFactory.name_registry:
                    if name_model in estimator_params.keys():
                        estimator_model[name_model] = ModelFactory.create_executor(name_model,
                                                                                   env = self.env,
                                                                                   env_config = self.env_config,
                                                                                   config = self.config,
                                                                                   **estimator_params[name_model])
                    else:
                        estimator_model[name_model] = ModelFactory.create_executor(name_model,
                                                                                   env = self.env,
                                                                                   env_config = self.env_config,
                                                                                   config = self.config,
                                                                                   )
        else:
            for name_model in estimator:
                if name_model in estimator_params.keys():
                    estimator_model[name_model] = ModelFactory.create_executor(name_model,
                                                                               env = self.env,
                                                                               env_config = self.env_config,
                                                                               config = self.config,
                                                                               **estimator_params[name_model])
                else:
                    estimator_model[name_model] = ModelFactory.create_executor(name_model,
                                                                               env = self.env,
                                                                               env_config = self.env_config,
                                                                               config = self.config,
                                                                               )

        # update estimator_params
        for name_model, params in estimator_params.items():
            self.estimator_params[name_model] = params

        for name_model, model in estimator_model.items():
            print('name_model: ', name_model)
            try:
                estimator = model.estimator
            except:
                estimator = model
            estimator.fit()
            results = estimator.predict()
            name_model = ''.join(name_model.split('-')[1:])

            self.metrics[name_model] = results['reward']
        return self
    def report_rl(self):
        print('report: ', self.metrics)



