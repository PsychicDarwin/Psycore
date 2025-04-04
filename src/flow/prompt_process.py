from typing import Dict, Optional, Any
from src.flow.processflow import ProcessFlow
from src.rl.q_learning import QLearningModel, ORIGINAL_PROMPT, ELABORATED_PROMPT

class PromptProcess(ProcessFlow):
    def __init__(self, q_learning_model: QLearningModel, prompt_elaborater, type: str = None, next: 'ProcessFlow' = None):
        """
        Initialize the PromptProcess.
        
        Args:
            q_learning_model: The Q-learning model for selecting prompts
            prompt_elaborater: Function/class to elaborate prompts
            type: Process type identifier
            next: Next process in the chain
        """
        self.prompt_elaborater = prompt_elaborater
        self.q_learning_model = q_learning_model
        self.next = next
        self.type = type
        self.original_prompt = None
        self.elaborated_prompt = None
        self.original_response = None
        self.elaborated_response = None
        self.selected_prompt = None
    
    def run(self, data: Dict) -> Dict:
        """
        Run the prompt process.
        
        Args:
            data: Input data containing at least an "initial_prompt" key
            
        Returns:
            Dictionary containing processed results
        """
        # When running a chat process we expect data in the form
        # {
        #     "initial_prompt" : str
        #     # may include an image attachment
        # }
        
        # Save the original prompt
        self.original_prompt = data["initial_prompt"]
        
        # Check if the data includes an image
        has_image = "image" in data or any(
            isinstance(val, dict) and val.get("type") == "image" 
            for val in data.values() if isinstance(val, dict)
        )
        
        # We elaborate the initial prompt with the prompt elaborater (An LLM)
        self.elaborated_prompt = self.prompt_elaborater(self.original_prompt)
        
        # Save both prompts in the output
        output = {
            "original_prompt": self.original_prompt,
            "elaborated_prompt": self.elaborated_prompt
        }
        
        # Determine if we should get both responses or choose one based on the model
        if data.get("exploration_mode", False):
            # In exploration mode, we'll provide both answers to the user
            # so they can choose which they prefer
            output["mode"] = "exploration"
            output["response_options"] = {
                "original": None,  # Will be populated by the next process
                "elaborated": None  # Will be populated by the next process
            }
            
            # If there's a next process, we'll run both prompts through it
            if self.next is not None:
                # Get response for original prompt
                original_input = data.copy()
                original_input["prompt"] = self.original_prompt
                original_input["prompt_type"] = "original"
                original_response = self.next.run(original_input)
                
                # Get response for elaborated prompt
                elaborated_input = data.copy()
                elaborated_input["prompt"] = self.elaborated_prompt
                elaborated_input["prompt_type"] = "elaborated"
                elaborated_response = self.next.run(elaborated_input)
                
                # Store both responses
                self.original_response = original_response.get("response", "No response")
                self.elaborated_response = elaborated_response.get("response", "No response")
                
                # Add responses to output
                output["response_options"] = {
                    "original": self.original_response,
                    "elaborated": self.elaborated_response
                }
                
                # We're returning both options for the user to choose from
                return output
        else:
            # In normal mode, use Q-learning to select the prompt
            output["mode"] = "exploitation" 
            
            # Use Q-learning model to decide which prompt to use
            action = self.q_learning_model.select_action(self.original_prompt, has_image)
            self.selected_prompt = self.original_prompt if action == ORIGINAL_PROMPT else self.elaborated_prompt
            output["selected_prompt"] = self.selected_prompt
            output["selected_type"] = "original" if action == ORIGINAL_PROMPT else "elaborated"
            
            # If there's a next process in the chain
            if self.next is not None:
                # Pass the selected prompt to the next process
                next_input = data.copy()
                next_input["prompt"] = self.selected_prompt
                next_input["prompt_type"] = output["selected_type"]
                response = self.next.run(next_input)
                output.update(response)
        
        return output
    
    def process_feedback(self, data: Dict) -> None:
        """
        Process user feedback to update the Q-learning model.
        
        Args:
            data: Dictionary containing feedback information
        """
        if "preferred_response" not in data:
            raise ValueError("Feedback data must include 'preferred_response'")
        
        preferred = data["preferred_response"]
        
        # Update the Q-learning model with user preference
        self.q_learning_model.process_feedback(preferred)
