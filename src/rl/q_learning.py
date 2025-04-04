"""
Q-Learning implementation for prompt optimization.

This module provides a Q-learning model that learns whether to use 
original or elaborated prompts based on user feedback.
"""

import numpy as np
import os
import json
import pickle
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants for actions
ORIGINAL_PROMPT = 0
ELABORATED_PROMPT = 1

class QLearningModel:
    """
    Q-learning model for learning which prompt type (original or elaborated)
    performs better for different types of queries.
    """
    
    def __init__(self, 
                learning_rate: float = 0.1, 
                discount_factor: float = 0.9,
                exploration_rate: float = 0.3,
                exploration_decay: float = 0.995,
                min_exploration_rate: float = 0.05,
                save_path: str = "results/q_learning"):
        """
        Initialize the Q-learning model.
        
        Args:
            learning_rate: Alpha parameter - how quickly to learn from new experiences
            discount_factor: Gamma parameter - importance of future rewards
            exploration_rate: Epsilon parameter - probability of trying random actions
            exploration_decay: Factor to multiply exploration_rate after each update
            min_exploration_rate: Minimum exploration rate
            save_path: Directory to save Q-table and metrics
        """
        self.q_table = {}  # Maps state keys to action values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.save_path = save_path
        
        # For tracking current state and action
        self.current_state_key = None
        self.current_action = None
        
        # For tracking metrics
        self.stats = {
            "total_updates": 0,
            "original_selected": 0,
            "elaborated_selected": 0,
            "history": []
        }
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Try to load existing Q-table
        self._load_q_table()
    
    def extract_features(self, prompt: str, has_image: bool = False) -> Dict[str, Any]:
        """
        Extract features from a prompt to create its state representation.
        
        Args:
            prompt: The prompt text
            has_image: Whether the prompt includes an image
            
        Returns:
            Dictionary of features
        """
        # Convert to lowercase for easier matching
        text = prompt.lower()
        
        # Basic features
        features = {
            "has_image": has_image,
            "length_category": "short" if len(text) < 20 else ("medium" if len(text) < 50 else "long"),
            "is_question": "?" in text,
        }
        
        # Categorize the prompt type based on keywords
        if any(word in text for word in ["explain", "describe", "what", "tell me"]):
            features["prompt_type"] = "descriptive"
        elif any(word in text for word in ["compare", "difference", "versus", "vs"]):
            features["prompt_type"] = "comparative"
        elif any(word in text for word in ["how to", "steps", "procedure", "instructions"]):
            features["prompt_type"] = "procedural"
        elif any(word in text for word in ["why", "reason", "because"]):
            features["prompt_type"] = "causal"
        elif any(word in text for word in ["analyze", "examine", "investigate"]):
            features["prompt_type"] = "analytical"
        elif any(word in text for word in ["list", "enumerate"]):
            features["prompt_type"] = "listing"
        else:
            features["prompt_type"] = "general"
        
        # Estimate complexity based on prompt structure
        # Here we're using a simple measure - we could make this more sophisticated
        words = text.split()
        features["complexity"] = "simple" if len(words) < 10 else ("medium" if len(words) < 20 else "complex")
        
        return features
    
    def _get_state_key(self, features: Dict[str, Any]) -> str:
        """
        Convert features dictionary to a string key for the Q-table.
        
        Args:
            features: Dictionary of state features
            
        Returns:
            String key representing the state
        """
        # Create a sorted string of key-value pairs
        # This ensures consistent key generation regardless of dict order
        state_parts = []
        for key, value in sorted(features.items()):
            state_parts.append(f"{key}:{value}")
        
        return "|".join(state_parts)
    
    def select_action(self, prompt: str, has_image: bool = False) -> int:
        """
        Select an action (which prompt to use) based on the current state.
        
        Args:
            prompt: The original prompt text
            has_image: Whether the prompt includes an image
            
        Returns:
            Action index (0 for original prompt, 1 for elaborated prompt)
        """
        # Extract features and get state key
        features = self.extract_features(prompt, has_image)
        state_key = self._get_state_key(features)
        self.current_state_key = state_key
        
        # Exploration: choose randomly
        if np.random.random() < self.exploration_rate:
            action = np.random.choice([ORIGINAL_PROMPT, ELABORATED_PROMPT])
            logger.info(f"Exploring: Randomly selected action {action}")
        else:
            # Exploitation: choose best action
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0, 0.0]  # [original, elaborated]
            
            q_values = self.q_table[state_key]
            
            # Choose action with highest Q-value
            if q_values[ORIGINAL_PROMPT] > q_values[ELABORATED_PROMPT]:
                action = ORIGINAL_PROMPT
            elif q_values[ELABORATED_PROMPT] > q_values[ORIGINAL_PROMPT]:
                action = ELABORATED_PROMPT
            else:
                # If Q-values are equal, choose randomly
                action = np.random.choice([ORIGINAL_PROMPT, ELABORATED_PROMPT])
            
            logger.info(f"Exploiting: Selected action {action} based on Q-values {q_values}")
        
        # Save the selected action
        self.current_action = action
        
        # Update selection counters
        if action == ORIGINAL_PROMPT:
            self.stats["original_selected"] += 1
        else:
            self.stats["elaborated_selected"] += 1
        
        return action
    
    def update(self, reward: float) -> None:
        """
        Update Q-values based on the received reward.
        
        Args:
            reward: The reward value (positive for good outcomes, negative for bad)
        """
        if self.current_state_key is None or self.current_action is None:
            logger.warning("Cannot update Q-values: no action has been selected yet")
            return
        
        # Get current Q-value
        if self.current_state_key not in self.q_table:
            self.q_table[self.current_state_key] = [0.0, 0.0]
        
        current_q = self.q_table[self.current_state_key][self.current_action]
        
        # Simple Q-value update (no next state in this scenario)
        # Q(s,a) = Q(s,a) + α * (r - Q(s,a))
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        # Update Q-table
        self.q_table[self.current_state_key][self.current_action] = new_q
        
        logger.info(f"Updated Q-value for state {self.current_state_key}, action {self.current_action}")
        logger.info(f"  Old Q-value: {current_q}, New Q-value: {new_q}")
        
        # Update exploration rate (decay)
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
        # Update statistics
        self.stats["total_updates"] += 1
        self.stats["history"].append({
            "timestamp": datetime.now().isoformat(),
            "state_key": self.current_state_key,
            "action": "original" if self.current_action == ORIGINAL_PROMPT else "elaborated",
            "reward": reward,
            "new_q_value": new_q,
            "exploration_rate": self.exploration_rate
        })
        
        # Save periodically
        if self.stats["total_updates"] % 10 == 0:
            self.save()
    
    def process_feedback(self, preferred_prompt_type: str) -> None:
        """
        Process user feedback about which prompt type they preferred.
        
        Args:
            preferred_prompt_type: String indicating which prompt was preferred
                                  ("original" or "elaborated")
        """
        if preferred_prompt_type not in ["original", "elaborated"]:
            logger.warning(f"Invalid preferred_prompt_type: {preferred_prompt_type}")
            return
        
        # Convert string to action constant
        preferred_action = ORIGINAL_PROMPT if preferred_prompt_type == "original" else ELABORATED_PROMPT
        
        # Calculate reward based on whether the preferred action matches our selection
        if preferred_action == self.current_action:
            reward = 1.0  # Positive reward if we selected the preferred prompt
        else:
            reward = -1.0  # Negative reward if we selected the wrong prompt
        
        # Update Q-values with this reward
        self.update(reward)
        
        logger.info(f"Processed feedback: User preferred {preferred_prompt_type}, reward = {reward}")
    
    def save(self) -> None:
        """Save the Q-table and stats to disk."""
        # Save Q-table
        q_table_path = os.path.join(self.save_path, "q_table.pkl")
        with open(q_table_path, "wb") as f:
            pickle.dump(self.q_table, f)
        
        # Save stats (but limit history to last 100 entries to keep file manageable)
        stats_for_save = self.stats.copy()
        if len(stats_for_save["history"]) > 100:
            stats_for_save["history"] = stats_for_save["history"][-100:]
        
        stats_path = os.path.join(self.save_path, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_for_save, f, indent=2, default=str)
        
        logger.info(f"Saved Q-learning model to {self.save_path}")
    
    def _load_q_table(self) -> None:
        """Attempt to load a saved Q-table from disk."""
        q_table_path = os.path.join(self.save_path, "q_table.pkl")
        stats_path = os.path.join(self.save_path, "stats.json")
        
        # Load Q-table if it exists
        if os.path.exists(q_table_path):
            try:
                with open(q_table_path, "rb") as f:
                    self.q_table = pickle.load(f)
                logger.info(f"Loaded Q-table with {len(self.q_table)} states")
                
                # Load stats if they exist
                if os.path.exists(stats_path):
                    with open(stats_path, "r") as f:
                        self.stats = json.load(f)
                    logger.info(f"Loaded stats: {self.stats['total_updates']} updates recorded")
            except Exception as e:
                logger.error(f"Error loading saved Q-learning model: {e}")
                self.q_table = {}  # Start fresh if loading fails
    
    def get_q_value(self, state_key: str, action: int) -> float:
        """
        Get the Q-value for a specific state-action pair.
        
        Args:
            state_key: The state key string
            action: The action (0 for original, 1 for elaborated)
            
        Returns:
            The Q-value
        """
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key][action]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for the Q-learning model.
        
        Returns:
            Dictionary of statistics
        """
        stats_copy = self.stats.copy()
        
        # Calculate percentages
        total_selections = stats_copy["original_selected"] + stats_copy["elaborated_selected"]
        if total_selections > 0:
            stats_copy["original_percentage"] = (stats_copy["original_selected"] / total_selections) * 100
            stats_copy["elaborated_percentage"] = (stats_copy["elaborated_selected"] / total_selections) * 100
        else:
            stats_copy["original_percentage"] = 0
            stats_copy["elaborated_percentage"] = 0
        
        # Add current exploration rate
        stats_copy["exploration_rate"] = self.exploration_rate
        
        # Add Q-table summary
        stats_copy["q_table_size"] = len(self.q_table)
        
        return stats_copy
    
    def __call__(self, prompt_text: str, has_image: bool = False) -> int:
        """
        Make the model callable to easily select an action.
        This allows using the model like: action = model(prompt_text)
        
        Args:
            prompt_text: The original prompt text
            has_image: Whether the prompt includes an image
            
        Returns:
            The selected action (0 for original prompt, 1 for elaborated prompt)
        """
        return self.select_action(prompt_text, has_image)
