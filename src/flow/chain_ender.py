from src.flow.processflow import ProcessFlow

class ChainEnder('ProcessFlow'):
    def __init__(self, runner: object = None, next: 'ProcessFlow' = None):
        self.next = None
        self.runner = None # This is an object we're passing through, could be a model, but that depends on the child class

    # This will be overridden to devise our own chain operations like langchain
    # But in our custom implementation
    def run(self, data: dict = {}, chain_memory :dict =  {}) -> dict:
        # This is just a simple chain invocation, so data can be passed between a series of processes
        return chain_memory
    
    