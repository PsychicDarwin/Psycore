"""
Test cases for calculator operations.
"""
import pytest
from src.calculator.operations import add

def test_add():
    """Test the add function with various inputs."""
    # Test positive numbers
    assert add(1, 2) == 3
    
    # Test negative numbers
    assert add(-1, -1) == -2
    
    # Test zero
    assert add(0, 5) == 5
    
    # Test floating point numbers
    assert add(0.1, 0.2) == pytest.approx(0.3)  # Using approx for floating point comparison