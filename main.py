import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# add labels
fake["label"] = 0   # fake = 0
true["label"] = 1   # real = 1

# combine data
data = pd.concat([fake, true])

# shuffle data
data = data.sample(frac=1, random_state=42)

# use only text + label
data = data[["text", "label"]]

print(data["label"].value_counts())

# split input/output
X = data["text"]
y = data["label"]

# convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_vector = vectorizer.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

# accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# user input
while True:
    user_input = input("Enter news: ")
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)

    if prediction[0] == 0:
        print("This news is FAKE ❌")
    else:
        print("This news is REAL ✅")