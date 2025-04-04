import streamlit as st
import pickle
from PIL import Image

# Load model/vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load images
banner = Image.open("banner.jpg")
real_img = Image.open("true.png")
fake_img = Image.open("fake.png")

# UI
st.image(banner, use_column_width=True)
st.title("🧠 AI-Powered Fake News Detector")
st.write("Enter a news article or statement to check if it's real or fake:")

input_text = st.text_area("Enter news content here", height=200)

if st.button("🔍 Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some content.")
    else:
        vect_text = vectorizer.transform([input_text])
        prediction = model.predict(vect_text)[0]

        if prediction == 1:
            st.image(real_img, width=300)
            st.success("✅ This news seems REAL!")
        else:
            st.image(fake_img, width=300)
            st.error("🚫 This news seems FAKE!")
