import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_path = "model"
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
    return model.eval(), tokenizer

model, tokenizer = load_model()

st.title("🍷 Wine Score Predictor")
text = st.text_area("Please enter a wines details or a brief review:")

if st.button("Predict"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            score = logits.squeeze().item()
            st.success(f"Predicted score/class: {score:.2f}")
    else:
        st.warning("Please enter some text.")


