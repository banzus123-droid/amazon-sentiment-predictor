import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import joblib
from textblob import TextBlob  # For detecting score from text

# --- INITIALIZATION ---
# Ensure NLTK data is available
# Force download all required resources to prevent LookupErrors on the cloud
nltk.download('punkt')
nltk.download('punkt_tab')  # Critical for word_tokenize in new NLTK versions
nltk.download('stopwords')
nltk.download('wordnet')

# Now import the components that depend on those downloads
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- PREPROCESSING FUNCTIONS ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_model_and_preprocessing():
    """Loads the trained model and preprocessing objects."""
    model = joblib.load('sentiment_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('feature_selector.pkl')
    
    # Load the preprocessed data to get clipping thresholds
    df = pd.read_csv("amazon_orange_readable.csv")
    rl_upper = df["review_length"].quantile(0.99)
    # Load score from raw dataset
    df_raw = pd.read_csv("amazon_review_dataset.csv")
    score_upper = df_raw["score"].quantile(0.99)
    
    return model, tfidf, selector, scaler, rl_upper, score_upper

# --- UI IMPLEMENTATION ---
st.set_page_config(page_title="Amazon Review Sentiment Predictor", page_icon="🛒")

st.title("🛒 Amazon Sentiment Predictor")
st.markdown("""
Predict whether a customer review is **Positive** or **Negative** using machine learning.
This tool analyzes the text content, automatically detects a star rating, and considers review length.
""")

# Model Explanation
st.subheader("📖 How It Works")
st.markdown("""
Our system analyzes reviews using three key factors:
- 📝 **Review Text**: The words you write (positive words = good sentiment!)
- ⭐ **Review Score**: The system automatically infers a star rating (1-5) from your text
- 📏 **Review Length**: How long your review is (balanced length works best)

💡 **Tip**: The system detects the star rating based on the sentiment in your words—no manual input needed!
""")

# Example Section
st.subheader("🔍 Example")
with st.expander("Click to see an example review"):
    st.markdown("""
    **Sample Review:** "This product exceeded my expectations! Great quality and fast shipping."
    
    **Score:** 5 stars (inferred from positive language)
    
    **Predicted Sentiment:** Positive 😊
    
    *Try entering this example to see how it works!*
    """)

# Load components
with st.spinner("Loading model and preprocessing objects..."):
    model, tfidf, selector, scaler, rl_upper, score_upper = load_model_and_preprocessing()

# Sidebar for Info
st.sidebar.header("Model Information")
st.sidebar.info("The model used is **Logistic Regression** trained on TF-IDF features with Chi-Square feature selection, combined with normalized numerical features (review score and review length).")

# User Inputs
st.subheader("✏️ Enter Your Review Details")
user_review = st.text_area("Review Text", placeholder="Example: I really loved this product, it worked perfectly!", height=150)

# Prediction Logic
if st.button("Analyze Sentiment", type="primary"):
    if not user_review.strip():
        st.warning("Please enter some review text first.")
    else:
        # 1. Clean Text
        cleaned = clean_text(user_review)
        rev_len = len(cleaned.split())
        
        # 2. Detect Score from Text
        polarity = TextBlob(cleaned).sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
        detected_score = int(round((polarity + 1) * 2)) + 1  # Map polarity to 1-5 stars
        detected_score = max(1, min(5, detected_score))  # Ensure within 1-5
        
        # 3. Process Numerical Features
        rl_clipped = min(rev_len, rl_upper)
        score_clipped = min(detected_score, score_upper)
        num_features = scaler.transform([[np.log1p(score_clipped), np.log1p(rl_clipped)]])
        
        # 3. Process Text Features
        text_tfidf = tfidf.transform([cleaned])
        text_selected = selector.transform(text_tfidf)
        
        # 4. Final Features
        final_input = hstack([text_selected, num_features])
        
        # 5. Predict
        prediction = model.predict(final_input)[0]
        if detected_score <= 3:
            prediction = 0
        
        # 6. Display Output
        st.divider()
        if prediction == 1:
            st.success("### Result: POSITIVE 😊")
            st.balloons()
        else:
            st.error("### Result: NEGATIVE ☹️")
        
        # Show key inputs used
        st.info(f"**Key Inputs Processed:** ⭐ Score: {score_clipped} | 📏 Word Count: {rev_len}")
        
        with st.expander("View Full Processing Details"):
            st.write(f"**Cleaned Text:** {cleaned}")
            st.write(f"**Score:** {detected_score} (polarity: {polarity:.2f}, clipped at 99th percentile: {score_upper:.0f})")
            st.write(f"**Model Used:** Logistic Regression")

st.markdown("---")

# Tips Section
st.subheader("💡 Quick Tips")
st.markdown("""
- **Write naturally**: The system understands everyday language and detects sentiment from your words.
- **System detects score**: No need to input stars—the system infers them automatically.
- **Length is key**: Too short or too long reviews might confuse the system.
- **Experiment**: Try writing positive or negative reviews to see how the detected score changes!
""")


st.caption("A251 STISK 2133 Predictive Analytics - Assignment 2")

