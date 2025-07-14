import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Load and prepare dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2. Add extra spam examples
extra_spam = [
    "Earn money fast! No investment needed!",
    "You’ve won ₹10,000 cash! Claim now!",
    "Work from home & get rich. Limited spots.",
    "WIN a free laptop! Sign up today!",
    "Lowest loan rates! Apply instantly!",
    "Congratulations! You've been selected for a cash prize.",
    "Apply for a free credit card today and get ₹500 instantly!",
    "Get rich in 7 days by joining our crypto scheme now!",
    "Cheap loans available instantly, no documents needed.",
    "Urgent! Your account will be blocked. Login and verify immediately.",
    "Your PayTM KYC is pending. Click now to update and avoid suspension."
]

extra_spam += [
    "Win a FREE iPhone now! Click the link to claim!",
    "Congratulations, you have won a lottery of $10,000!",
    "Click here to reset your password immediately!",
    "URGENT: Your account is locked. Verify now to avoid suspension.",
    "Get exclusive deals only for you! Act fast.",
    "You're selected for a free vacation trip to Dubai!",
    "Buy 1 get 1 FREE! Limited stock available now!",
    "Invest ₹500 and earn ₹50,000 in 7 days! Limited offer!",
    "Update your KYC or your PayTM account will be blocked.",
    "Hot singles are waiting for you nearby. Chat now!",
    "Loan approved! Instant disbursal without documents!",
    "Get Viagra at 90% off! Only for today!",
    "Your OTP is 123456. Do not share with anyone.",
    "Congratulations! You've been selected for a Google Survey reward.",
    "You won't believe this weight loss trick. Doctors hate it!"
]
extra_df = pd.DataFrame({'text': extra_spam, 'label': [1] * len(extra_spam)})
df = pd.concat([df, extra_df], ignore_index=True)

# 3. Train-test split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Create individual models
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

nb = MultinomialNB()

# 6. Ensemble Voting Classifier (hard voting)
ensemble_model = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('nb', nb)],
    voting='soft'
)

# 7. Train the ensemble
ensemble_model.fit(X_train_vec, y_train)

# 8. Evaluate
y_pred = ensemble_model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Save model and vectorizer
joblib.dump(ensemble_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Final ensemble model saved.")
