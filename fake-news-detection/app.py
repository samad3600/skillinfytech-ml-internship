import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------- UI --------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")
st.write("This app uses Machine Learning (NLP) to classify news as **Real or Fake**.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This project uses TF-IDF and Logistic Regression to detect fake news.\n\n"
    "Built using Streamlit + Scikit-learn."
)

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    return data.sample(frac=1).reset_index(drop=True)

data = load_data()

# -------------------- Clean Text --------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data["text"] = data["text"].apply(clean_text)

# -------------------- Train Model --------------------
@st.cache_resource
def train_model():
    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vect = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vect, y)

    return model, vectorizer

model, vectorizer = train_model()

# -------------------- Input --------------------
news_input = st.text_area("✍️ Enter News Text:")

col1, col2 = st.columns(2)

# -------------------- Predict Button --------------------
with col1:
    if st.button("🔍 Predict"):
        if news_input:
            cleaned = clean_text(news_input)
            vect = vectorizer.transform([cleaned])
            result = model.predict(vect)
            proba = model.predict_proba(vect)

            confidence = max(proba[0]) * 100

            if result[0] == 1:
                st.success(f"✅ Real News ({confidence:.2f}% confidence)")
            else:
                st.error(f"❌ Fake News ({confidence:.2f}% confidence)")
        else:
            st.warning("Please enter some text!")

# -------------------- Sample Button --------------------
with col2:
    if st.button("💡 Try Sample"):
        sample = "Government announces new education reform policy"
        st.info(sample)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")