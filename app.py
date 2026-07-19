import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

X = data["text"]
y = data["label"]

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_vector = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vector, y)

# UI
st.title("📰 Fake News Detection App")
st.write("This app uses Machine Learning and NLP to detect whether news is Real or Fake.")

news = st.text_area("Enter News Here")

if st.button("Predict"):
    if news:
        vector = vectorizer.transform([news])

        with st.spinner("Analyzing news..."):
            result = model.predict(vector)
            proba = model.predict_proba(vector)

            st.write("Confidence:", round(proba.max()*100, 2), "%")

            if result[0] == 1:
                st.success("This is REAL news ✅")
            else:
                st.error("This is FAKE news ❌")

    else:
        st.warning("Please enter some text")
        st.markdown("---")
st.write("Developed by Charitha N C")
       