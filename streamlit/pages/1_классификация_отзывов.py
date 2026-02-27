import streamlit as st
import torch
import time
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

#########################################
#              ЗАГРУЗКА МОДЕЛЕЙ
#########################################

@st.cache_resource
def load_baseline():
    model = joblib.load("models/reviews/baseline_model.pkl")
    with open("models/reviews/tokenizer.json") as f:
        tok = json.load(f)
    vectorizer = TfidfVectorizer(**tok["params"])
    vectorizer.vocabulary_ = tok["vocab"]
    return model, vectorizer


@st.cache_resource
def load_lstm():
    # Загрузка словаря
    with open("models/reviews/vocab_LSTM.json") as f:
        vocab = json.load(f)

    # Обратный словарь
    word2idx = vocab

    # Модель
    model = torch.load("models/reviews/lstm_sentiment.pt", map_location="cpu")
    model.eval()
    return model, word2idx


def lstm_encode(text, word2idx, max_len=200):
    tokens = text.lower().split()
    ids = [word2idx.get(t, word2idx.get("<unk>", 1)) for t in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return torch.tensor([ids])


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("models/reviews/bert_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/reviews/bert_model"
    )
    model.eval()
    return tokenizer, model


#########################################
#              ИНТЕРФЕЙС
#########################################

st.title("🩺 Sentiment Classification — ML → LSTM → BERT")

text = st.text_area("Введите отзыв пациента:", height=150)

if not text.strip():
    st.stop()

#########################################
#          BASELINE (LogReg)
#########################################

baseline_model, baseline_vec = load_baseline()

t0 = time.time()
X = baseline_vec.transform([text])
pred = baseline_model.predict(X)[0]
baseline_time = (time.time() - t0) * 1000

st.subheader("Baseline (LogReg • TF-IDF)")
st.write(f"Предсказание: **{pred}**")
st.write(f"Время инференса: {baseline_time:.2f} ms")


#########################################
#                  LSTM
#########################################

lstm_model, word2idx = load_lstm()

ids = lstm_encode(text, word2idx)
t0 = time.time()
with torch.no_grad():
    logits = lstm_model(ids)
    probs = torch.softmax(logits, dim=1)
    pred_lstm = torch.argmax(probs, dim=1).item()
lstm_time = (time.time() - t0) * 1000

st.subheader("LSTM (classic RNN)")
st.write(f"Предсказание: **{pred_lstm}**")
st.write(f"Время инференса: {lstm_time:.2f} ms")


#########################################
#                 BERT
#########################################

bert_tok, bert_model = load_bert()

inputs = bert_tok(text, return_tensors="pt", truncation=True, max_length=256)

t0 = time.time()
with torch.no_grad():
    outputs = bert_model(**inputs, output_attentions=True)
bert_time = (time.time() - t0) * 1000

probs = F.softmax(outputs.logits, dim=1)
pred_bert = torch.argmax(probs, dim=1).item()

st.subheader("BERT (Transformer)")
st.write(f"Предсказание: **{pred_bert}**")
st.write(f"Время инференса: {bert_time:.2f} ms")


#########################################
#         Attention Visualization
#########################################

att = outputs.attentions[-1][0]    # последний слой вниманий, head=0
att = att.mean(dim=0)               # среднее по heads → [seq, seq]

tokens = bert_tok.convert_ids_to_tokens(inputs["input_ids"][0])

st.markdown("### Attention Heatmap")

df_att = pd.DataFrame(att[:len(tokens), :len(tokens)].numpy(),
                      index=tokens, columns=tokens)

st.dataframe(df_att.style.background_gradient(cmap="viridis"))


#########################################
#         Сводная таблица F1
#########################################

try:
    with open("models/reviews/metrics_LSTM_review.json") as f:
        m_lstm = json.load(f)
    with open("models/reviews/baseline_results.json") as f:
        m_base = json.load(f)
    with open("models/reviews/bert_model/bert_metrics_BERT.json") as f:
        m_bert = json.load(f)

    df = pd.DataFrame({
        "Model": ["Baseline", "LSTM", "BERT"],
        "F1-macro": [
            m_base["f1_macro"],
            m_lstm["f1_macro"],
            m_bert["f1_macro"]
        ]
    })

    st.markdown("### 📊 Сводная таблица F1-macro")
    st.table(df)

except Exception as e:
    st.write("Не удалось загрузить метрики:", str(e))