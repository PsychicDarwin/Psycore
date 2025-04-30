"""
Script to test structured output support for all models in the catalogue.
"""
import sys
import os
from pathlib import Path
import subprocess
import time
import csv
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from typing import List
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.model.model_catalogue import ModelCatalogue, ModelType, LocalModelType
from src.model.wrappers import ChatModelWrapper

class TestEntities(BaseModel):
    """
    Simple test model for structured output.
    """
    names: List[str] = Field(
        ...,
        description="List of names extracted from the text",
    )

def download_ollama_model(model_name: str) -> bool:
    """
    Download an Ollama model if it's not already present.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Check if model is already downloaded
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if model_name in result.stdout:
            print(f"Model {model_name} already downloaded")
            return True
        
        # Download the model
        print(f"Downloading {model_name}...")
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor download progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return process.returncode == 0
        
    except Exception as e:
        print(f"Error downloading {model_name}: {str(e)}")
        return False

def test_model(model_type: ModelType) -> bool:
    """
    Test if a model supports structured output.
    
    Args:
        model_type: The model type to test
        
    Returns:
        bool: True if the model supports structured output, False otherwise
    """
    try:
        # Skip models over 100GB
        if isinstance(model_type, LocalModelType) and model_type.download_size and model_type.download_size > 100:
            print(f"⚠️ Skipping {model_type.argName} - download size {model_type.download_size}GB exceeds 100GB limit")
            return False
            
        # Download Ollama models if needed
        if model_type.provider.name == "OLLAMA":
            if not download_ollama_model(model_type.argName):
                print(f"❌ Failed to download {model_type.argName}")
                return False
        
        # Initialize the model
        model_wrapper = ChatModelWrapper(model_type)
        model = model_wrapper.model
        
        # Create a simple prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a test model."),
            ("human", "Extract names from: John and Jane went to the park."),
        ])
        
        # Try to use structured output
        chain = prompt | model.with_structured_output(TestEntities)
        result = chain.invoke({"question": "John and Jane went to the park."})
        
        # If we get here, it worked
        print(f"✅ {model_type.argName} ({model_type.provider.name}) - Success!")
        return True
        
    except Exception as e:
        # If we get an error, it didn't work
        print(f"❌ {model_type.argName} ({model_type.provider.name}) - Failed: {str(e)}")
        return False

def save_results_to_csv(results: dict, output_dir: str = "results"):
    """
    Save test results to a CSV file.
    
    Args:
        results: Dictionary containing test results
        output_dir: Directory to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_test_results_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Write results to CSV
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['name', 'model', 'provider', 'supports_structured_output', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for name, result in results.items():
            row = {
                'name': name,
                'model': result['model'],
                'provider': result['provider'],
                'supports_structured_output': result['supports_structured_output'],
                'timestamp': timestamp
            }
            writer.writerow(row)
    
    print(f"\nResults saved to: {filepath}")

def main():
    """
    Test all models in the catalogue for structured output support.
    """
    print("Testing structured output support for all models...\n")
    
    # Get all models
    all_models = ModelCatalogue._models
    
    # Test each model
    results = {}
    for name, model_type in all_models.items():
        print(f"\nTesting {name} ({model_type.argName})...")
        results[name] = {
            "model": model_type.argName,
            "provider": model_type.provider.name,
            "supports_structured_output": test_model(model_type)
        }
    
    # Print summary
    print("\n=== Summary ===")
    working_models = [name for name, result in results.items() if result["supports_structured_output"]]
    failed_models = [name for name, result in results.items() if not result["supports_structured_output"]]
    
    print(f"\nWorking models ({len(working_models)}):")
    for name in working_models:
        print(f"- {name}: {results[name]['model']} ({results[name]['provider']})")
    
    print(f"\nFailed models ({len(failed_models)}):")
    for name in failed_models:
        print(f"- {name}: {results[name]['model']} ({results[name]['provider']})")
    
    # Save results to CSV
    save_results_to_csv(results)

if __name__ == "__main__":
    main() 