# adapters/protocol.py
from abc import ABC, abstractmethod

class NeuralAdapter(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def apply_stimulus(self, stim):
        pass

    @abstractmethod
    def run(self, duration_ms):
        pass

    @abstractmethod
    def get_output(self):
        pass
