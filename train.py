import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import os
import re
from scipy.sparse import hstack

def extract_additional_features(df):
    """Extract additional engineered features from the email data"""
    # Email length features
    df['email_length'] = df['email_text'].apply(len)
    df['subject_length'] = df['subject'].apply(len)
    
    # Link features
    df['link_density'] = df['links_count'] / (df['email_length'] + 1)
    
    # Domain features
    df['domain_age'] = df['sender_domain'].apply(lambda x: hash(x) % 30)  # Placeholder for actual domain age
    
    # Special character features
    df['special_chars'] = df['email_text'].apply(lambda x: len(re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', x)))
    
    # HTML tag features
    df['html_tags'] = df['email_text'].apply(lambda x: len(re.findall(r'<[^>]+>', x.lower())))
    
    return df

def train_and_save_model():
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Convert labels to binary (1 for phishing, 0 for legitimate)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'phishing' else 0)
    
    # Extract additional features
    df = extract_additional_features(df)
    
    # Split data into features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipeline
    text_transformer = Pipeline([
        ('hashing', HashingVectorizer(n_features=2**16, alternate_sign=False,
                                     stop_words='english', ngram_range=(1, 2))),
    ])
    
    categorical_transformer = Pipeline([
        ('hash', HashingVectorizer(n_features=100, alternate_sign=False))
    ])
    
    numeric_features = ['has_attachment', 'links_count', 'urgent_keywords',
                       'email_length', 'subject_length', 'link_density',
                       'domain_age', 'special_chars', 'html_tags']
    
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('email_text', text_transformer, 'email_text'),
            ('subject', text_transformer, 'subject'),
            ('sender_domain', categorical_transformer, 'sender_domain'),
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Create and train the model - using GradientBoosting for better accuracy
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8,
            max_features='sqrt'
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/phishing_detection_model.pkl', compress=3)
    print("Model saved as 'models/phishing_detection_model.pkl'")

if __name__ == '__main__':
    train_and_save_model()