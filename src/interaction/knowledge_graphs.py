from typing import List, Tuple, Union
from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.model.model_catalogue import ModelType
from src.model.wrappers import ChatModelWrapper

# It's a simple implementation, but it immediately plugs any custom model into the LLMGraphTransformer
class ModelKGTransformer(LLMGraphTransformer):
    def __init__(self, model: ChatModelWrapper, allowed_nodes: List[str] = [],allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = []):
        super().__init__(llm = model.model, allowed_nodes = allowed_nodes, allowed_relationships = allowed_relationships)

    def __init__(self, modelType: ModelType, allowed_nodes: List[str] = [],allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = []):
        super().__init__(llm = ChatModelWrapper(modelType).model, allowed_nodes = allowed_nodes, allowed_relationships = allowed_relationships)
