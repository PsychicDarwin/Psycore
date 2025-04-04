"""
Demo script for Q-learning prompt selection.

This script demonstrates how to use the Q-learning model to select between
original and elaborated prompts based on user feedback.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.flow.prompt_elaborator import PromptElaborator
from src.rl.q_learning import QLearningModel, ORIGINAL_PROMPT, ELABORATED_PROMPT
from src.model.model_catalogue import ModelCatalogue
from src.model.wrappers import ChatModelWrapper

def main():
    """Run the Q-learning demo."""
    print("=== Q-Learning Prompt Selection Demo ===\n")
    
    # Initialize the prompt elaborator with an LLM
    print("Initializing prompt elaborator...")
    elaborator = PromptElaborator(model_name="claude_3_sonnet")
    
    # Initialize the Q-learning model
    print("Initializing Q-learning model...")
    q_model = QLearningModel(
        learning_rate=0.1,
        exploration_rate=0.3,  # Start with high exploration
        save_path="results/q_learning_demo"
    )
    
    # Initialize a model for testing
    print("Initializing test model...")
    multimodal_models = ModelCatalogue.get_MLLMs()
    test_model_name = "claude_3_sonnet" if "claude_3_sonnet" in multimodal_models else list(multimodal_models.keys())[0]
    print(f"Using model: {test_model_name}")
    test_model = ChatModelWrapper(multimodal_models[test_model_name]).model
    
    # Run demo loop
    while True:
        print("\n" + "="*50)
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        print("\nElaborating prompt...")
        elaborated_prompt = elaborator(prompt)
        
        print("\nOriginal prompt:")
        print(prompt)
        
        print("\nElaborated prompt:")
        print(elaborated_prompt)
        
        # Select action using Q-learning (which prompt to use)
        action = q_model.select_action(prompt)
        selected_prompt = prompt if action == ORIGINAL_PROMPT else elaborated_prompt
        prompt_type = "Original" if action == ORIGINAL_PROMPT else "Elaborated"
        
        print(f"\nQ-learning selected: {prompt_type} prompt")
        
        # Get both responses for comparison
        print("\nGetting responses for both prompts...")
        
        try:
            original_response = test_model.invoke(prompt)
            elaborated_response = test_model.invoke(elaborated_prompt)
            
            print("\nResponse to original prompt:")
            print(original_response.content[:500] + "..." if len(original_response.content) > 500 else original_response.content)
            
            print("\nResponse to elaborated prompt:")
            print(elaborated_response.content[:500] + "..." if len(elaborated_response.content) > 500 else elaborated_response.content)
            
            # Get user feedback
            while True:
                feedback = input("\nWhich response was better? (o)riginal, (e)laborated, or (n)o preference: ").lower()
                if feedback in ['o', 'original']:
                    q_model.process_feedback("original")
                    print("Recorded preference for original prompt")
                    break
                elif feedback in ['e', 'elaborated']:
                    q_model.process_feedback("elaborated") 
                    print("Recorded preference for elaborated prompt")
                    break
                elif feedback in ['n', 'none', 'no preference']:
                    print("No preference recorded")
                    break
                else:
                    print("Invalid input, please try again")
            
            # Show updated stats
            stats = q_model.get_stats()
            print("\nCurrent Q-learning stats:")
            print(f"Total updates: {stats['total_updates']}")
            print(f"Original prompt selections: {stats['original_selected']} ({stats['original_percentage']:.1f}%)")
            print(f"Elaborated prompt selections: {stats['elaborated_selected']} ({stats['elaborated_percentage']:.1f}%)")
            print(f"Current exploration rate: {stats['exploration_rate']:.3f}")
            
        except Exception as e:
            print(f"Error getting responses: {e}")
    
    # Save the Q-learning model before exiting
    q_model.save()
    print("\nQ-learning model saved. Demo complete.")

if __name__ == "__main__":
    main()
