
  # Machine Learning Based Support Ticket Classification System
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
# Dataset
tickets = [
    "Internet is not working",
    "Unable to login to account",
    "Payment failed during transaction",
    "Website loading very slow",
    "Forgot my password",
    "Application crashes frequently",
    "Need refund for payment",
    "Cannot connect to WiFi",
    "Account hacked please help",
    "Order delivery delayed"
]
# Categories
categories = [
    "Technical",
    "Account",
    "Payment",
    "Technical",
    "Account",
    "Technical",
    "Payment",
    "Technical",
    "Security",
    "Delivery"
]
# Priority Function
def assign_priority(ticket):
    ticket = ticket.lower()
    if "hacked" in ticket or "failed" in ticket or "crashes" in ticket:
        return "High"
    elif "slow" in ticket or "delayed" in ticket or "forgot" in ticket:
        return "Medium"
    else:
        return "Low"
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    tickets,
    categories,
    test_size=0.3,
    random_state=42
)
# Machine Learning Pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
# Train model
model.fit(X_train, y_train)
# Predictions
predictions = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("\n===== MODEL ACCURACY =====")
print("Accuracy :", round(accuracy * 100, 2), "%")
# User Input
new_ticket = input("\nEnter Support Ticket : ")
# Predict category
predicted_category = model.predict([new_ticket])[0]
# Predict priority
priority = assign_priority(new_ticket)
# Final Output
print("\n===== RESULT =====")
print("Ticket Category :", predicted_category)
print("Priority Level  :", priority)
