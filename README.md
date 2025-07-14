# SMS Spam Detection using Machine Learning

This project aims to detect spam SMS messages using machine learning techniques. An ensemble model combining Logistic Regression, Random Forest, and Multinomial Naive Bayes is trained on labeled SMS data to classify messages as spam or not spam. A Streamlit web app is developed for real-time prediction.

## Features

- Real-time spam classification of SMS messages
- Ensemble model with high accuracy (98.75%)
- Simple web interface built using Streamlit
- Includes training script, dataset, and saved model files

## Dataset

- Source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Total messages: 5,574 (ham and spam)
- Additional synthetic spam messages included to improve class balance

## Technologies Used

- Python 3.10+
- Scikit-learn
- Pandas
- Joblib
- Streamlit
- CountVectorizer (Bag-of-Words)
