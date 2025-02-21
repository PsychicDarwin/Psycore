from abc import ABC, abstractmethod

class ProcessFlow(ABC):
    def __init__(self, data: dict = None, runner: object = None, next: 'ProcessFlow' = None):
        self.next = next
        self.data = data
        self.runner = runner # This is an object we're passing through, could be a model, but that depends on the child class

    # This will be overridden to devise our own chain operations like langchain
    # But in our custom implementation
    @abstractmethod
    def run(self):
        return self.data