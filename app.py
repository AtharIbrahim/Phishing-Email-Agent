from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import re

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
        'urgent_keywords': 0
    }
    
    # Update with provided values
    if has_attachment is not None:
        features['has_attachment'] = int(has_attachment)
    
    # Count links if not provided
    if links_count is None and email_text:
        # Simple link counting (matches http/https URLs)
        features['links_count'] = len(re.findall(r'https?://\S+', email_text))
    elif links_count is not None:
        features['links_count'] = int(links_count)
    
    # Extract domain from email text if not provided
    if sender_domain is None and email_text:
        # Simple domain extraction (looks for @domain patterns)
        domain_match = re.search(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_text)
        if domain_match:
            features['sender_domain'] = domain_match.group(1)
    
    # Detect urgent keywords if not provided
    if urgent_keywords is None and email_text:
        urgent_phrases = ['urgent', 'immediate', 'action required', 'verify now', 'security alert']
        features['urgent_keywords'] = 1 if any(phrase in email_text.lower() for phrase in urgent_phrases) else 0
    elif urgent_keywords is not None:
        features['urgent_keywords'] = int(urgent_keywords)
    
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
            'probability': float(probability[0][1] if prediction[0] == 1 else float(probability[0][0])),
            'confidence': float(max(probability[0])),
            'features_used': features  # Return the actual features used
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # Get form data
        email_text = request.form.get('email_text', '')
        subject = request.form.get('subject', '')
        has_attachment = request.form.get('has_attachment', 0)
        links_count = request.form.get('links_count', 0)
        sender_domain = request.form.get('sender_domain', "testing.com")
        urgent_keywords = request.form.get('urgent_keywords', 0)
        
        # Extract features with smart defaults
        features = extract_features_from_email(
            email_text=email_text,
            subject=subject,
            has_attachment=has_attachment,
            links_count=links_count,
            sender_domain=sender_domain,
            urgent_keywords=urgent_keywords
        )
        
        # Create input data
        input_data = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        result = {
            'prediction': 'phishing' if prediction[0] == 1 else 'legitimate',
            'probability': float(probability[0][1] if prediction[0] == 1 else probability[0][0]),
            'confidence': float(max(probability[0])),
            'features_used': features
        }
        
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your account has been compromised. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Can you review my document?\"}"



# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout. Click here to secure it!\", \"subject\": \"Urgent: Account Security Alert\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Monthly newsletter - May Edition\", \"subject\": \"Company Newsletter\", \"has_attachment\": 0, \"links_count\": 0, \"sender_domain\": \"company.com\", \"urgent_keywords\": 0}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Reset your password now to avoid lockout.\", \"subject\": \"Password Reset Required\", \"has_attachment\": 0, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"Your Netflix account has been suspended.\", \"subject\": \"Netflix Account Notice\", \"has_attachment\": 1, \"links_count\": 2, \"sender_domain\": \"security-alert.com\", \"urgent_keywords\": 1}"
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"email_text\": \"our account is on hold. Log in now to avoid suspension.\"}"