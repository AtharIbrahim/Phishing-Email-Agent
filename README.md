# 🛡️ PhishShield – Advanced Phishing Detection System

PhishShield is a powerful AI-driven phishing detection agent that analyzes emails and web content to identify potential phishing threats in real-time. Using an intelligently trained **Random Forest** classifier on a dataset of **200k+ emails**, PhishShield leverages linguistic patterns, link analysis, and structural metadata to predict whether a message is phishing or legitimate with over **96% of  accuracy**.    

# 📌 Overview

PhishShield consists of

- **🧠 Backend (Flask API):** ML-powered REST API for real-time predictions.
- **🎨 Frontend (HTML/CSS/JS):** Interactive UI for email inspection and manual inputs.
- **📊 ML Model:** Pre-trained model using advanced feature engineering.
- **📁 Dataset:** 200,000 labeled email samples with 7 rich features for training and research..


### Model Working

<div align="center">
  <img src="https://github.com/AtharIbrahim/Phishing-Email-Agent/blob/main/screenshots/working.png" height="400" alt="Main Interface">
</div>



# 🌟 Key Features

### 🚀 Real-time Detection

- REST API via Flask backend
- JSON-based predictions with confidence scores
- Auto model training fallback (`train.py`)

### 🧪 Smart Feature Engineering

- 25+ extracted features including:
  - Domain reputation
  - Link patterns
  - Urgent language indicators
  - HTML tag frequency
  - Attachment behavior

### 🧠 ML Model Highlights

- **Random Forest Classifier**
- Over **96% accuracy**
- Trained on 200k labeled samples
- Explainable predictions

### 💡 User-Friendly Interface

- Drag-and-drop email file analysis (.eml, .msg)
- Manual content entry support
- Visual risk indicator (safe/suspicious)
- Historical detection log
- Responsive UI for all devices

---
# PhishShield Demo Walkthrough


### Main Dashboard
<img src="https://github.com/AtharIbrahim/Phishing-Email-Agent/blob/main/screenshots/frontend1.png" width="800" alt="Main Interface">

**Features:**
- Dual input methods (file upload + text input)
- Real-time analysis results
- Confidence visualization


### Manual Analysis Mode
<img src="https://github.com/AtharIbrahim/Phishing-Email-Agent/blob/main/screenshots/frontend2.png" width="800" alt="Manual Input">

**Workflow:**
1. Paste email content
2. Optionally add metadata
3. Get instant phishing probability


### File Upload Analysis
<img src="https://github.com/AtharIbrahim/Phishing-Email-Agent/blob/main/screenshots/frontend3.png" width="800" alt="File Upload">

**Supports:**
- `.eml` and `.msg` email files
- Automatic header parsing
- Attachment detection


### Analysis History
<img src="https://github.com/AtharIbrahim/Phishing-Email-Agent/blob/main/screenshots/frontend4.png" width="800" alt="History Panel">

**Key Functions:**
- Persistent scan records
- Quick-result filtering
- One-click reanalysis

---

# 📊 Dataset

PhishShield includes a **robust and reusable dataset** of **200,000+ labeled emails** for training, evaluation, and experimentation.

### 📑 Labels (Features):
| Feature            | Description |
|--------------------|-------------|
| `email_text`       | Body content of the email |
| `subject`          | Email subject line |
| `has_attachment`   | Binary flag (1 = yes, 0 = no) |
| `links_count`      | Number of hyperlinks detected |
| `sender_domain`    | Domain of sender’s email address |
| `urgent_keywords`  | Binary flag (1 = urgent words found) |
| `label`            | Target class: `phishing` or `legitimate` |

> 🧠 Ideal for building and enhancing phishing classifiers or integrating into broader cybersecurity AI pipelines.

---

## 🛠️ Installation & Setup

### 📋 Prerequisites
- Python 3.8+
- Git
- Node.js (for frontend development, optional)

### 🔧 Backend Setup
```bash
# Clone the repository
git clone https://github.com/AtharIbrahim/Phishing-Email-Agent.git
cd Phishing-Email-Agent

# Create virtual environment
python -m venv venv

# Activate environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py

```

## 📬 Contact
<ul>
  <li><strong>Name</strong>: Athar Ibrahim Khalid</li>
  <li><strong>GitHub</strong>: <a href="https://github.com/AtharIbrahim/" target="_blank">https://github.com/AtharIbrahim/</a></li>
  <li><strong>LinkedIn</strong>: <a href="https://www.linkedin.com/in/athar-ibrahim-khalid-0715172a2/" target="_blank">LinkedIn Profile</a></li>
  <li><strong>Website</strong>: <a href="https://atharibrahimkhalid.netlify.app/" target="_blank">Athar Ibrahim Khalid</a></li>
</ul>


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
