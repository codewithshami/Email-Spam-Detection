# 📧 Email Spam Detection Web App

This is a machine learning-based **Spam Mail Classifier** built using `scikit-learn`, `SMOTE`, and `Streamlit`. It predicts whether an email or message is **Spam** or **Ham (Not Spam)** based on its content.

---

## 🚀 Features

- Logistic Regression-based classifier
- Text vectorization using TF-IDF (with n-grams)
- SMOTE balancing to handle imbalanced datasets
- Real-time prediction via a user-friendly Streamlit interface
- Confusion matrix and classification report for evaluation
- Dataset distribution chart (Ham vs Spam)

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib (for pie chart)

---

## 📂 Dataset

The app uses a labeled dataset (`mail_data.csv`) containing spam and ham messages.

- Column `Category`: `spam` or `ham`
- Column `Message`: the actual email/message text

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier


🖥️ App Preview

✍️ Example Spam Messages for Testing
Congratulations! You've won a free iPhone. Click here to claim now!

Your account will be deactivated. Log in now to avoid suspension.

Get a ₹50,000 loan approved instantly!

📊 Model Evaluation
Accuracy

Confusion Matrix

Classification Report

Pie chart showing class distribution
