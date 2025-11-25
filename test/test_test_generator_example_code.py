"""Example functions for testing the test generator"""

def add_numbers(a, b):
    """Add two numbers"""
    return a + b

def calculate_average(numbers):
    """Calculate average of a list"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def greet_user(name):
    """Greet a user"""
    return f"Hello, {name}!"

def is_even(num):
    """Check if number is even"""
    return num % 2 == 0

print("Example functions loaded")
