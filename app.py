from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import re
from urllib.parse import urlparse
import tldextract
import numpy as np

app = Flask(__name__)

# Load the model at startup
model = None
def load_model():
    global model
    if not os.path.exists('models/phishing_detection_model.pkl'):
        print("Model not found. Training a new one...")
        from train import train_and_save_model
        train_and_save_model()
    model = joblib.load('models/phishing_detection_model.pkl')

load_model()

def extract_domain_features(domain):
    """Extract features from domain name"""
    if not domain:
        return {
            'domain_length': 0,
            'subdomain_count': 0,
            'hyphen_count': 0,
            'digit_count': 0
        }
    
    extracted = tldextract.extract(domain)
    main_domain = f"{extracted.domain}.{extracted.suffix}"
    
    return {
        'domain_length': len(main_domain),
        'subdomain_count': len(extracted.subdomain.split('.')),
        'hyphen_count': main_domain.count('-'),
        'digit_count': sum(c.isdigit() for c in main_domain)
    }

def extract_features_from_email(email_text, subject="", has_attachment=None, links_count=None, sender_domain=None, urgent_keywords=None):
    """
    Extract features from available email data with smart defaults for missing values
    """
    # Default values
    features = {
        'email_text': email_text if email_text else "",
        'subject': subject if subject else "",
        'has_attachment': 0,
        'links_count': 0,
        'sender_domain': "",
        'urgent_keywords': 0,
        # Additional features with defaults
        'email_length': 0,
        'subject_length': 0,
        'link_density': 0,
        'domain_age': 0,
        'special_chars': 0,
        'html_tags': 0
    }
    
    # Update with provided values
    if has_attachment is not None:
        features['has_attachment'] = int(has_attachment)
    
    # Count links if not provided
    if links_count is None and email_text:
        # More robust link counting
        links = re.findall(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', email_text.lower())
        features['links_count'] = len(links)
    elif links_count is not None:
        features['links_count'] = int(links_count)
    
    # Extract domain from email text if not provided
    if sender_domain is None and email_text:
        # Improved domain extraction
        domain_match = re.search(
            r'[\w\.-]+@([\w\.-]+\.\w{2,})|https?://([\w\.-]+\.\w{2,})', 
            email_text.lower())
        if domain_match:
            features['sender_domain'] = domain_match.group(1) or domain_match.group(2)
    
    # Detect urgent keywords if not provided
    if urgent_keywords is None and email_text:
        urgent_phrases = ['urgent', 'immediate', 'action required', 'verify now', 
                         'security alert', 'account suspended', 'password expired',
                         'click here', 'limited time', 'offer expires']
        features['urgent_keywords'] = int(any(phrase in email_text.lower() for phrase in urgent_phrases))
    elif urgent_keywords is not None:
        features['urgent_keywords'] = int(urgent_keywords)
    
    # Calculate additional features
    features['email_length'] = len(features['email_text'])
    features['subject_length'] = len(features['subject'])
    features['link_density'] = features['links_count'] / (features['email_length'] + 1)
    
    # Domain features
    domain_features = extract_domain_features(features['sender_domain'])
    features.update(domain_features)
    
    # Special characters
    features['special_chars'] = len(re.findall(r'[!$%^&*()_+|~=`{}\[\]:";\'<>?,./]', features['email_text']))
    
    # HTML tags
    features['html_tags'] = len(re.findall(r'<[^>]+>', features['email_text'].lower()))
    
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features with smart defaults
        features = extract_features_from_email(
            email_text=data.get('email_text', ''),
            subject=data.get('subject', ''),
            has_attachment=data.get('has_attachment'),
            links_count=data.get('links_count'),
            sender_domain=data.get('sender_domain'),
            urgent_keywords=data.get('urgent_keywords')
        )
        
        # Create a DataFrame with the features
        input_data = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Prepare response
        response = {
            'prediction': 'phishing' if prediction[0] == 1 else 'legitimate',
            'probability': float(probability[0][1]),
            'confidence': float(max(probability[0])),
            'features_used': {
                k: v for k, v in features.items() 
                if k not in ['email_text', 'subject']  # Exclude large text fields
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your account has been compromised. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Can you review my document?\"}"



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Final notice: subscription expired.\", \"subject\": \"Unusual Login Attempt\", \"has_attachment\": 1, \"links_count\": 2, \"sender_domain\": \"travelprizes.org\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Monthly newsletter - May Edition\", \"subject\": \"Company Newsletter\", \"has_attachment\": 0, \"links_count\": 0, \"sender_domain\": \"company.com\", \"urgent_keywords\": 0}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout.\", \"subject\": \"Password Reset Required\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your Netflix account has been suspended.\", \"subject\": \"Netflix Account Notice\", \"has_attachment\": 1, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"our account is on hold. Log in now to avoid suspension.\"}"