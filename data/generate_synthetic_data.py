import numpy as np
import pandas as pd
import datetime
import random

def generate_synthetic_data(n_samples=1000, fraud_ratio=0.1):
    np.random.seed(42)
    
    # Generate customer IDs
    n_customers = int(n_samples * 0.3)  # Some customers have multiple orders
    customer_ids = [f'CUST_{i:05d}' for i in range(n_customers)]
    
    # Generate order data
    data = {
        'customer_id': np.random.choice(customer_ids, n_samples),
        'order_id': [f'ORDER_{i:06d}' for i in range(n_samples)],
        'claim_id': [f'CLAIM_{i:06d}' for i in range(n_samples)],
        'product_category': np.random.choice(['Electronics', 'Groceries', 'Clothing', 'Home', 'Beauty'], n_samples),
        'total_order_value': np.random.uniform(100, 5000, n_samples),
        'refund_amount': np.random.uniform(50, 2000, n_samples),
        'claim_reason': np.random.choice(['Expired', 'Damaged', 'Wrong Item', 'Missing Parts', 'Not as Described'], n_samples)
    }
    
    # Generate timestamps
    base_date = datetime.datetime(2023, 1, 1)
    
    # Order timestamps
    order_timestamps = [base_date + datetime.timedelta(days=np.random.randint(0, 365)) 
                       for _ in range(n_samples)]
    data['order_timestamp'] = order_timestamps
    
    # Delivery timestamps (1-5 days after order)
    data['delivery_timestamp'] = [ts + datetime.timedelta(days=np.random.randint(1, 6)) 
                                for ts in order_timestamps]
    
    # Claim timestamps (0-30 days after delivery for legitimate, 25-40 for fraudulent)
    fraud_flags = np.random.choice([0, 1], n_samples, p=[1-fraud_ratio, fraud_ratio])
    
    claim_timestamps = []
    for i in range(n_samples):
        if fraud_flags[i] == 0:  # Legitimate
            delay = np.random.randint(0, 31)
        else:  # Fraudulent
            delay = np.random.randint(25, 41)
        claim_timestamps.append(data['delivery_timestamp'][i] + datetime.timedelta(days=delay))
    
    data['claim_timestamp'] = claim_timestamps
    data['fraud_flag'] = fraud_flags
    
    # Generate claim descriptions
    legitimate_templates = [
        "The product was {issue} when I received it. I would like a refund.",
        "I received the item but it was {issue}. Please refund.",
        "Unfortunately the {product} I ordered was {issue} on arrival.",
        "The {product} doesn't work as expected because it's {issue}.",
        "I'm disappointed that the {product} came {issue}."
    ]
    
    fraudulent_templates = [
        "The product is {issue}. I want my money back immediately.",
        "Your {product} is completely {issue}. I demand a refund now.",
        "This is the third time I've received a {issue} {product}. Refund!!!",
        "I can't believe how {issue} this {product} is. I want a full refund.",
        "This {product} is the worst I've ever bought, totally {issue}."
    ]
    
    issues = ['damaged', 'broken', 'expired', 'defective', 'not as described']
    
    descriptions = []
    for i in range(n_samples):
        product = data['product_category'][i].lower()
        issue = np.random.choice(issues)
        
        if fraud_flags[i] == 0:  # Legitimate
            template = np.random.choice(legitimate_templates)
        else:  # Fraudulent
            template = np.random.choice(fraudulent_templates)
            
        descriptions.append(template.format(product=product, issue=issue))
    
    data['claim_description'] = descriptions
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df