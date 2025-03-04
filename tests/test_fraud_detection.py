import pytest
from refund_fraud_detection import RefundFraudDetectionSystem
from data.generate_synthetic_data import generate_synthetic_data

def test_preprocess_data():
    fraud_detector = RefundFraudDetectionSystem()
    df = generate_synthetic_data(100)
    processed_data = fraud_detector.preprocess_data(df)
    
    assert 'claim_hour' in processed_data.columns
    assert 'claim_day' in processed_data.columns
    assert 'days_since_order' in processed_data.columns

def test_train_model():
    fraud_detector = RefundFraudDetectionSystem()
    df = generate_synthetic_data(100)
    processed_data = fraud_detector.preprocess_data(df)
    
    exclude_cols = ['fraud_flag', 'customer_id', 'order_id', 'claim_id', 'claim_timestamp', 
                   'order_timestamp', 'delivery_timestamp', 'claim_description']
    
    feature_cols = [col for col in processed_data.columns if col not in exclude_cols]
    
    X_train = processed_data[feature_cols]
    y_train = processed_data['fraud_flag']
    
    fraud_detector.train_models(X_train, y_train)
    
    assert fraud_detector.classifier is not None
    assert fraud_detector.anomaly_detector is not None