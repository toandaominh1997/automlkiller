from abc import ABCMeta, abstractmethod

class PreprocessFactory(object):
    registry = {}
    name_registry = []
    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class):
            if name in cls.registry:
                print('Executor {} already axists. Will replace it'.format(name))
            cls.registry[name] = wrapped_class
            cls.name_registry.append(name)
            return wrapped_class
        return inner_wrapper
    @classmethod
    def create_executor(cls, name, **kwargs):
        if name not in cls.registry:
            print('Executor {} does not exist in the registry'.format(name))
            return None
        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor
