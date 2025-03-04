from refund_fraud_detection import RefundFraudDetectionSystem
import joblib

def train_model(train_processed):
    # Initialize the fraud detection system
    fraud_detector = RefundFraudDetectionSystem()
    
    # Define features and target
    exclude_cols = ['fraud_flag', 'customer_id', 'order_id', 'claim_id', 'claim_timestamp', 
                   'order_timestamp', 'delivery_timestamp', 'claim_description']
    
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed['fraud_flag']
    
     # Debug: Check X_train columns
    # print("Columns in X_train:", X_train.columns.tolist())
    # Extract text features
    text_features_train = fraud_detector.engineer_features(train_processed)
    
    # Train models
    fraud_detector.train_models(X_train, y_train, text_features_train)
    
    # Save the model
    fraud_detector.save_model("models/refund_fraud_detector.joblib")
    
    return fraud_detector