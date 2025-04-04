from typing import Dict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.model.model_catalogue import ModelCatalogue, ModelType
from src.model.wrappers import ChatModelWrapper

class PromptElaborator:
    """
    A class that uses an LLM to elaborate on and improve basic prompts.
    This can enhance the effectiveness of prompts by adding context, instructions,
    or better formatting before sending to the main LLM.
    """
    
    def __init__(self, model_name: str = "claude_3_sonnet", save_history: bool = True):
        """
        Initialize the PromptElaborator with a specific model.
        
        Args:
            model_name: The name of the model to use for elaboration (must exist in ModelCatalogue)
            save_history: Whether to save elaboration history
        """
        # Get the model from ModelCatalogue
        model_type = ModelCatalogue._models.get(model_name)
        if not model_type:
            raise ValueError(f"Model '{model_name}' not found in ModelCatalogue")
        
        # Create the model wrapper
        self.model_wrapper = ChatModelWrapper(model_type)
        self.model = self.model_wrapper.model
        
        # Create the elaboration prompt template
        self.elaboration_template = PromptTemplate(
            input_variables=["original_prompt"],
            template="""
            You are an expert prompt engineer. Your task is to elaborate and improve the following prompt 
            to make it more effective for a language model. Add specific instructions, context, and formatting 
            that will help the model provide a better response.

            Original prompt: {original_prompt}

            Provide an elaborated version of this prompt that will get better results from an LLM. 
            Only return the improved prompt text, nothing else.
            """
        )
        
        # Create the chain
        self.elaboration_chain = LLMChain(
            llm=self.model,
            prompt=self.elaboration_template
        )
        
        self.save_history = save_history
        self.history = []
    
    def __call__(self, prompt: str) -> str:
        """
        Elaborate on a prompt using the LLM.
        
        Args:
            prompt: The original prompt to elaborate on
            
        Returns:
            The elaborated prompt
        """
        # Call the elaboration chain
        elaborated_prompt = self.elaboration_chain.run(original_prompt=prompt)
        
        # Clean up the elaborated prompt (remove quotes if present)
        elaborated_prompt = elaborated_prompt.strip()
        if elaborated_prompt.startswith('"') and elaborated_prompt.endswith('"'):
            elaborated_prompt = elaborated_prompt[1:-1]
        
        # Save to history if enabled
        if self.save_history:
            self.history.append({
                "original": prompt,
                "elaborated": elaborated_prompt
            })
        
        return elaborated_prompt
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the history of prompt elaborations."""
        return self.history
    
    def clear_history(self):
        """Clear the elaboration history."""
        self.history = []
