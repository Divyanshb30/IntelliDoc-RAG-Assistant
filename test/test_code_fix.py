"""
Shopping Cart System - Demo with intentional bugs
"""

import hashlib

# Issue 1: Mutable default argument
def add_items_to_cart(items=[]):
    items.append("new_item")
    return items

# Issue 2: Bare except clause
def calculate_total(prices):
    try:
        total = sum(prices)
        return total
    except:
        return 0

# Issue 3: Unused variable
def apply_discount(price, discount_rate):
    original_price = price
    tax = 0.08
    discounted = price * (1 - discount_rate)
    return discounted

# Issue 4: Weak cryptography (MD5)
def hash_user_password(password):
    return hashlib.md5(password.encode()).hexdigest()

# Issue 5: Deep nesting (3+ levels)
def process_orders(orders):
    result = []
    for order in orders:
        if order['status'] == 'pending':
            for item in order['items']:
                if item['in_stock']:
                    for discount in item['discounts']:
                        if discount > 0:
                            result.append(item)
    return result

# Issue 6: Missing docstring
def validate_email(email):
    return '@' in email and '.' in email

# Issue 7: Long line (100+ characters)
def generate_user_report(user_id, user_name, user_email, user_age, user_address, user_city, user_state, user_zip):
    return f"User {user_id}: {user_name} ({user_email}) - {user_age} years old - Address: {user_address}, {user_city}, {user_state} {user_zip}"

# Main execution
if __name__ == "__main__":
    cart = add_items_to_cart()
    print(f"Cart items: {cart}")
    
    prices = [10.99, 25.50, 15.00]
    total = calculate_total(prices)
    print(f"Total: ${total}")
    
    discounted = apply_discount(100, 0.2)
    print(f"Discounted price: ${discounted}")
    
    print("Demo complete!")
