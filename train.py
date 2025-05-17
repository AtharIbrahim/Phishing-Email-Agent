import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_and_save_model():
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Convert labels to binary (1 for phishing, 0 for legitimate)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'phishing' else 0)
    
    # Split data into features and target
    X = df[['email_text', 'subject', 'has_attachment', 'links_count', 'sender_domain', 'urgent_keywords']]
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing pipeline
    text_transformer = TfidfVectorizer(max_features=1000, stop_words='english')
    categorical_transformer = Pipeline([
        ('onehot', TfidfVectorizer(max_features=50))  # For sender_domain
    ])
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('email_text', text_transformer, 'email_text'),
            ('subject', text_transformer, 'subject'),
            ('sender_domain', categorical_transformer, 'sender_domain'),
            ('num', numeric_transformer, ['has_attachment', 'links_count', 'urgent_keywords'])
        ])
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['legitimate', 'phishing']))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/phishing_detection_model.pkl')
    print("Model saved as 'models/phishing_detection_model.pkl'")

if __name__ == '__main__':
    train_and_save_model()