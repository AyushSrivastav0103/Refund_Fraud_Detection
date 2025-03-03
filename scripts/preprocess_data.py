from refund_fraud_detection import RefundFraudDetectionSystem

def preprocess_data(train_df, test_df):
    # Initialize the fraud detection system
    fraud_detector = RefundFraudDetectionSystem()
    
    # Preprocess data
    train_processed = fraud_detector.preprocess_data(train_df)
    test_processed = fraud_detector.preprocess_data(test_df)
    
    return train_processed, test_processed