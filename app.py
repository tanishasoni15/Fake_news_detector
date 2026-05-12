import streamlit as st
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection")

input_news = st.text_area("Enter News Article")

if st.button("Predict"):

    if len(input_news) < 50:
        st.warning("Please enter a longer news article.")
    
    else:

        transformed_text = vectorizer.transform([input_news])

        prediction = model.predict(transformed_text)

        if prediction[0] == 1:
            st.success("Real News ✅")
        else:
            st.error("Fake News ❌")
