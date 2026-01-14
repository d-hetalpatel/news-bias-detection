import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="News Bias Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Headline Bias Classifier")
st.write(
    "Classify news headlines as **Left-leaning**, **Neutral**, or **Right-leaning** "
    "using a fine-tuned Transformer model."
)

# -------------------------------
# Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "models/news_bias_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

LABELS = ["left", "neutral", "right"]

# -------------------------------
# Prediction Function
# -------------------------------
def predict_bias(headline: str):
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    return LABELS[pred_id], probs[0][pred_id].item()

# -------------------------------
# UI Input
# -------------------------------
headline = st.text_area(
    "Enter a news headline:",
    placeholder="Government increases funding for renewable energy",
    height=100
)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Bias"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        label, confidence = predict_bias(headline)

        st.success(f"**Predicted Bias:** {label.upper()}")
        st.write(f"**Confidence:** {confidence:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using Transformers + Streamlit | NLP Project by ZenithIndia")




