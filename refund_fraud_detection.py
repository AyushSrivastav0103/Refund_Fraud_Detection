import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)


class RefundFraudDetectionSystem:
    """
    A comprehensive system for detecting potentially fraudulent refund claims
    using traditional machine learning approaches.
    """
    
    def __init__(self):
        # Initialize model components
        self.preprocessing_pipeline = None
        self.classifier = None
        self.anomaly_detector = None
        self.text_vectorizer = None
        self.feature_names = None
        self.threshold = 0.7  # Default threshold, can be tuned
        
    def preprocess_data(self, data):
        """
        Preprocess the raw data, including feature engineering
        
        Parameters:
        data (DataFrame): Raw data containing order and claim information
        
        Returns:
        DataFrame: Processed data with engineered features
        """
        if data is None:
            raise ValueError("Input data is None.")
        
        df = data.copy()
        
        # Temporal features
        df['claim_hour'] = df['claim_timestamp'].dt.hour
        df['claim_day'] = df['claim_timestamp'].dt.dayofweek
        df['claim_weekend'] = df['claim_day'].apply(lambda x: 1 if x >= 5 else 0)
        df['days_since_order'] = (df['claim_timestamp'] - df['order_timestamp']).dt.days
        df['days_since_delivery'] = (df['claim_timestamp'] - df['delivery_timestamp']).dt.days
        
        # If close to expiry date for applicable products
        if 'expiry_date' in df.columns:
            df['days_to_expiry'] = (df['expiry_date'] - df['claim_timestamp']).dt.days
            df['near_expiry'] = df['days_to_expiry'].apply(lambda x: 1 if x < 7 else 0)
        
        # Customer history features
        customer_stats = df.groupby('customer_id').agg({
            'order_id': 'count',
            'claim_id': 'count',
            'total_order_value': 'sum',
            'refund_amount': 'sum'
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'total_orders', 'total_claims', 
                                 'lifetime_order_value', 'lifetime_refund_value']
        
        customer_stats['claim_rate'] = customer_stats['total_claims'] / customer_stats['total_orders']
        customer_stats['refund_to_order_ratio'] = customer_stats['lifetime_refund_value'] / customer_stats['lifetime_order_value']
        
        # Merge customer stats back to main dataframe
        df = pd.merge(df, customer_stats, on='customer_id', how='left')
        
        # Product category features
        category_claim_rate = df.groupby('product_category').agg({
            'claim_id': 'count',
            'order_id': 'count'
        }).reset_index()
        
        category_claim_rate['category_claim_rate'] = category_claim_rate['claim_id'] / category_claim_rate['order_id']
        df = pd.merge(df, category_claim_rate[['product_category', 'category_claim_rate']], 
                     on='product_category', how='left')
        
        # Recent claim pattern
        recent_claims = df.sort_values('claim_timestamp').groupby('customer_id').agg({
            'claim_timestamp': lambda x: list(x)[-3:] if len(list(x)) >= 3 else list(x),
            'claim_reason': lambda x: list(x)[-3:] if len(list(x)) >= 3 else list(x)
        }).reset_index()
        
        # Calculate time between recent claims
        def calc_time_between_claims(timestamps):
            if len(timestamps) < 2:
                return [None]
            return [(timestamps[i] - timestamps[i-1]).days for i in range(1, len(timestamps))]
        
        recent_claims['days_between_claims'] = recent_claims['claim_timestamp'].apply(calc_time_between_claims)
        
        # Check for repeated same reason
        def check_repeated_reason(reasons):
            if len(reasons) < 2:
                return 0
            return 1 if len(set(reasons)) == 1 else 0
        
        recent_claims['repeated_reason'] = recent_claims['claim_reason'].apply(check_repeated_reason)
        
        # Calculate average time between claims
        def avg_time_between_claims(days_list):
            days_list = [d for d in days_list if d is not None]
            return np.mean(days_list) if days_list else None
        
        recent_claims['avg_days_between_claims'] = recent_claims['days_between_claims'].apply(avg_time_between_claims)
        
        # Merge back to main dataframe
        df = pd.merge(df, recent_claims[['customer_id', 'repeated_reason', 'avg_days_between_claims']], 
                     on='customer_id', how='left')
        
        # Fill NaN values in 'avg_days_between_claims' with 0
        df['avg_days_between_claims'] = df['avg_days_between_claims'].fillna(0)
        # Calculate z-score for claim timing pattern
        df['claim_timing_zscore'] = (df['avg_days_between_claims'] - df['avg_days_between_claims'].mean()) / df['avg_days_between_claims'].std()
        
        # Clean and preprocess text
        if 'claim_description' in df.columns:
            df['cleaned_description'] = df['claim_description'].apply(self._clean_text)
        
        # Fill NaN values in numerical columns with 0
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(0)

# Fill NaN values in categorical columns with 'unknown'
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        return df
    
    def _clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if w not in stop_words]
        
        return ' '.join(words)
    
    def engineer_features(self, df):
        """Additional feature engineering before model training"""
        
        # Extract categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target column and non-feature columns
        exclude_cols = ['fraud_flag', 'customer_id', 'order_id', 'claim_id', 'claim_timestamp', 
                         'order_timestamp', 'delivery_timestamp', 'claim_description']
        
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols and 'cleaned_' not in col]
        
        # Create preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessors
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        self.preprocessing_pipeline.fit(df)

        # Add text vectorization if text column exists
        if 'cleaned_description' in df.columns:
            self.text_vectorizer = TfidfVectorizer(max_features=100)
            text_features = self.text_vectorizer.fit_transform(df['cleaned_description'])
            
            # Store feature names for later use
            feature_names = (
                numerical_cols +
                self.text_vectorizer.get_feature_names_out().tolist() +
                self.preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
            )
            self.feature_names = feature_names
        
        else:
            feature_names = (
            numerical_cols +
            self.preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
        )
        self.feature_names = feature_names
    
        return text_features
    
    
    
    def train_models(self, X_train, y_train, text_features_train=None):
        """
        Train the classification and anomaly detection models
        
        Parameters:
        X_train (DataFrame): Training features
        y_train (Series): Training labels
        text_features_train (sparse matrix): Text features if available
        
        Returns:
        self: Trained model
        """
        # Apply preprocessing pipeline
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)
        
        # Combine with text features if available
        if text_features_train is not None:
            X_train_processed = np.hstack([X_train_processed.toarray(), text_features_train.toarray()])

        # Debug: Check for NaN or infinite values
        print("NaN values in X_train_processed:", np.isnan(X_train_processed).sum())
        print("Infinite values in X_train_processed:", np.isinf(X_train_processed).sum())
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
        
        # Train anomaly detector on legitimate claims only
        legitimate_indices = np.where(y_train == 0)[0]
        X_legitimate = X_train_processed[legitimate_indices]
        
        self.anomaly_detector = IsolationForest(
            contamination=0.05,  # Assuming 5% of "legitimate" claims might be anomalous
            random_state=42
        )
        self.anomaly_detector.fit(X_legitimate)
        
        # Train classifier
        self.classifier = xgb.XGBClassifier(
            scale_pos_weight=len(y_train) / sum(y_train),  # Additional class weight balance
            use_label_encoder=False,
            eval_metric='auc',
            random_state=42
        )
        
        # Grid search for hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200]
        }
        
        grid_search = GridSearchCV(
            self.classifier,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_resampled, y_resampled)
        self.classifier = grid_search.best_estimator_
        
        return self
    
    def evaluate_model(self, X_test, y_test, text_features_test=None):
        """
        Evaluate the model performance
        
        Parameters:
        X_test (DataFrame): Test features
        y_test (Series): Test labels
        text_features_test (sparse matrix): Text features if available
        
        Returns:
        dict: Performance metrics
        """
        # Apply preprocessing pipeline
        X_test_processed = self.preprocessing_pipeline.transform(X_test)
        
        # Combine with text features if available
        if text_features_test is not None:
            X_test_processed = np.hstack([X_test_processed.toarray(), text_features_test.toarray()])
        
        # Make predictions
        y_pred_proba = self.classifier.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        anomaly_scores = self.anomaly_detector.decision_function(X_test_processed)
        
        # Combine classifier and anomaly detector
        # Flag as fraud if either classifier confidence is high or anomaly score is very low
        combined_pred = np.logical_or(
            y_pred == 1,
            anomaly_scores < -0.5  # Threshold for strong anomalies
        ).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': (combined_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, combined_pred),
            'confusion_matrix': confusion_matrix(y_test, combined_pred)
        }
        
        print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Classification Report:\n{metrics['classification_report']}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Plot feature importance
        self._plot_feature_importance()
        
        return metrics
    
    def _plot_feature_importance(self):
        """Plot feature importance from the XGBoost classifier"""
        if not hasattr(self.classifier, 'feature_importances_'):
            return
        
        # Get feature importance
        importances = self.classifier.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Select top 15 features
        top_n = min(15, len(importances))
        
        # Display the feature ranking
        plt.figure(figsize=(10, 6))
        plt.title("Feature importances")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), np.array(self.feature_names)[indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.show()
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Parameters:
        data (DataFrame): New data to make predictions on
        
        Returns:
        DataFrame: Original data with fraud probability and flags
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Extract text features if available
        text_features = None
        if 'cleaned_description' in processed_data.columns and self.text_vectorizer is not None:
            text_features = self.text_vectorizer.transform(processed_data['cleaned_description'])
        
        # Apply preprocessing pipeline
        X_processed = self.preprocessing_pipeline.transform(processed_data)
        
        # Combine with text features if available
        if text_features is not None:
            X_processed = np.hstack([X_processed.toarray(), text_features.toarray()])
        
        # Get fraud probability from classifier
        fraud_proba = self.classifier.predict_proba(X_processed)[:, 1]
        
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        anomaly_scores = self.anomaly_detector.decision_function(X_processed)
        
        # Add predictions to original data
        result = data.copy()
        result['fraud_probability'] = fraud_proba
        result['anomaly_score'] = anomaly_scores
        
        # Flag as fraud if either classifier confidence is high or anomaly score is very low
        result['fraud_flag'] = np.logical_or(
            fraud_proba >= self.threshold,
            anomaly_scores < -0.5  # Threshold for strong anomalies
        ).astype(int)
        
        # Add risk tier
        def assign_risk_tier(row):
            if row['fraud_probability'] >= 0.8 or row['anomaly_score'] < -0.7:
                return 'High Risk'
            elif row['fraud_probability'] >= 0.5 or row['anomaly_score'] < -0.3:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        result['risk_tier'] = result.apply(assign_risk_tier, axis=1)
        
        return result
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        model_components = {
            'preprocessing_pipeline': self.preprocessing_pipeline,
            'classifier': self.classifier,
            'anomaly_detector': self.anomaly_detector,
            'text_vectorizer': self.text_vectorizer,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }
        joblib.dump(model_components, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        model_components = joblib.load(filepath)
        self.preprocessing_pipeline = model_components['preprocessing_pipeline']
        self.classifier = model_components['classifier']
        self.anomaly_detector = model_components['anomaly_detector']
        self.text_vectorizer = model_components['text_vectorizer']
        self.feature_names = model_components['feature_names']
        self.threshold = model_components['threshold']
        print(f"Model loaded from {filepath}")
        return self