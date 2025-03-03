from scripts.train_model import train_model
from scripts.predict import predict
from scripts.preprocess_data import preprocess_data
from data.generate_synthetic_data import generate_synthetic_data
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data(5000, 0.15)
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Preprocess data
    train_processed, test_processed = preprocess_data(train_df, test_df)
    
    # Train the model
    model = train_model(train_processed)
    
    # Make predictions
    predictions = predict(model, test_processed)
    
    # Display high-risk cases
    high_risk_cases = predictions[predictions['risk_tier'] == 'High Risk'].sort_values('fraud_probability', ascending=False)
    print("\nTop 5 high-risk cases:")
    print(high_risk_cases[['customer_id', 'claim_reason', 'fraud_probability', 'anomaly_score', 'risk_tier']].head())