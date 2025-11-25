"""
Sample Python file with various code issues for testing
This file intentionally contains bugs and security vulnerabilities
"""

# Issue 1: Division by zero risk
def calculate_average(numbers):
    """Calculate average without checking for empty list"""
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)  # BUG: Division by zero if empty!




# Issue 3: SQL Injection vulnerability
def get_user_by_id(user_id):
    """Fetch user from database - INSECURE!"""
    import sqlite3
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # VULNERABILITY: String formatting in SQL query
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    
    return cursor.fetchone()


# Issue 4: Bare except clause
def risky_operation():
    """Function with poor error handling"""
    try:
        result = 10 / 0
        return result
    except:  # BAD: Catches everything including KeyboardInterrupt
        pass


# Issue 5: Mutable default argument
def add_to_list(item, my_list=[]):
    """Classic Python gotcha - shared mutable default"""
    my_list.append(item)
    return my_list


# Issue 6: Using eval (CODE INJECTION RISK)
def calculate_expression(expr):
    """Evaluate user input - DANGEROUS!"""
    return eval(expr)  # CRITICAL: Can execute arbitrary code!


# Issue 7: Weak cryptography
import hashlib

def hash_password(password):
    """Hash password with MD5 - INSECURE!"""
    return hashlib.md5(password.encode()).hexdigest()  # Use bcrypt instead!


# Issue 8: File handling without context manager
def read_file(filename):
    """Read file without proper resource management"""
    f = open(filename, 'r')  # Should use 'with' statement
    content = f.read()
    # File might not close if exception occurs
    return content


# Issue 9: Deep nesting (code smell)
def complex_logic(data):
    """Overly complex nested conditions"""
    result = []
    for item in data:
        if item > 0:
            if item < 100:
                if item % 2 == 0:
                    if item % 3 == 0:  # 4 levels deep!
                        result.append(item)
    return result


# Issue 10: Long function (code smell)
def process_everything(data, config, options, flags):
    """
    This function does too many things
    Simulating a function over 50 lines
    """
    step1 = data * 2
    step2 = step1 + 10
    step3 = step2 - 5
    step4 = step3 / 2
    step5 = step4 * 3
    step6 = step5 + config
    step7 = step6 - options
    step8 = step7 * flags
    step9 = step8 + 100
    step10 = step9 - 50
    step11 = step10 * 2
    step12 = step11 / 4
    step13 = step12 + 25
    step14 = step13 - 10
    step15 = step14 * 3
    # ... imagine 40 more lines ...
    return step15


# Issue 11: Unused variable (potential)
def some_function():
    """Function with unused variables"""
    _temp_data = [1, 2, 3, 4, 5]
    _unused_var = "This might not be used"
    
    result = sum([1, 2, 3])
    return result


# Issue 12: Using pickle (deserialization risk)
import pickle

def load_data(filename):
    """Load pickled data - SECURITY RISK with untrusted data"""
    with open(filename, 'rb') as f:
        return pickle.load(f)  # Can execute arbitrary code!


# Issue 13: No input validation
def divide_numbers(a, b):
    """Divide without validation"""
    return a / b  # No check for b=0, no type checking


# Issue 14: Global variable mutation
counter = 0

def increment_counter():
    """Modify global state"""
    global counter  # Code smell: global state
    counter += 1


if __name__ == "__main__":
    print("This is a test file with intentional bugs!")
    print("Expected issues:")
    print("- Division by zero risks")
    print("- Hardcoded credentials (3)")
    print("- SQL injection")
    print("- Bare except")
    print("- Mutable defaults")
    print("- eval() usage")
    print("- Weak cryptography")
    print("- Resource leaks")
    print("- Deep nesting")
    print("- Long functions")
    print("- Unused variables")
    print("- Pickle usage")
    print("\nTotal: 12+ issues across various severity levels")
