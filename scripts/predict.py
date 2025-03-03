def predict(model, test_processed):
    # Define features
    exclude_cols = ['fraud_flag', 'customer_id', 'order_id', 'claim_id', 'claim_timestamp', 
                   'order_timestamp', 'delivery_timestamp', 'claim_description', 'cleaned_description']
    
    feature_cols = [col for col in test_processed.columns if col not in exclude_cols]
    
    X_test = test_processed[feature_cols]
    y_test = test_processed['fraud_flag']
    
    # Extract text features
    if 'cleaned_description' in test_processed.columns and model.text_vectorizer is not None:
        text_features_test = model.text_vectorizer.transform(test_processed['cleaned_description'])
    else:
        text_features_test = None
    
    # Evaluate models
    metrics = model.evaluate_model(X_test, y_test, text_features_test)
    
    # Make predictions on new data
    predictions = model.predict(test_processed)
    
    return predictions