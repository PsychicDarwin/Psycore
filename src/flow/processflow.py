from abc import ABC, abstractmethod

class ProcessFlow(ABC):
    def __init__(self, runner: object = None, next: 'ProcessFlow' = None):
        self.next = next
        self.runner = runner # This is an object we're passing through, could be a model, but that depends on the child class

    # This will be overridden to devise our own chain operations like langchain
    # But in our custom implementation
    @abstractmethod
    def run(self, data: dict, chain_memory = {}) -> dict:
        # This is just a simple chain invocation, so data can be passed between a series of processes
        chain_memory.update(data)
        if self.next is not None:
            return self.next.run(data)
        return data
    
    def change_next(self, next: 'ProcessFlow'):
        self.next = next
        return self
    
    