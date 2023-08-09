from abc import ABC, abstractmethod


class BaseModelFactory(ABC):

    def load(self, device_type):
        if device_type in ['cpu', 'CPU']:
            model = self.load_cpu_model()
        elif device_type in ['mps', 'MPS']:
            model = self.load_mps_model()
        else:
            model = self.load_cuda_model()
        return model

    @abstractmethod
    def load_cpu_model(self):
        pass

    @abstractmethod
    def load_mps_model(self):
        pass

    @abstractmethod
    def load_cuda_model(self):
        pass
