import streamlit as st
import torch
import time
import pandas as pd
import json
from utils import page_header

# === 🔥 ИНИЦИАЛИЗАЦИЯ SESSION_STATE ===
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "by_sentiment": {'Негативный': 0, 'Позитивный': 0},
        "by_model": {}
    }
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# === Конфигурация ===
SENTIMENT_LABELS = ['Негативный', 'Позитивный']
LABEL_MAPPING = {'Негативный': 0, 'Позитивный': 1}
ID_TO_LABEL = {0: 'Негативный', 1: 'Позитивный'}

MODEL_PATHS = {
    "Baseline (LogReg)": "Lesia-Alba/nlp_reviews_baseline",
    "LSTM": "Lesia-Alba/nlp_reviews_lstm",
    "BERT (ai-forever)": "Lesia-Alba/nlp_reviews_bert"
}

st.set_page_config(page_title="Отзывы", page_icon="🏥", layout="wide")
page_header("Классификация отзывов на поликлиники", "🏥")

# === Загрузка модели ===
@st.cache_resource
def load_model_cached(model_name, repo_id):
    from huggingface_hub import hf_hub_download
    import torch

    try:
        # === LOGREG ===
        if "LogReg" in model_name:
            import joblib

            pkl_path = hf_hub_download(
                repo_id=repo_id,
                filename="baseline_model.pkl"
            )

            model = joblib.load(pkl_path)

            return {"type": "logreg", "model": model}, None

        # === LSTM ===
        elif "LSTM" in model_name:
            from lstm_model import LSTMClassifier
            import json

            # загружаем три файла
            cfg_path = hf_hub_download(repo_id=repo_id, filename="lstm_config.json")
            vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab_LSTM.json")
            weights_path = hf_hub_download(repo_id=repo_id, filename="best_lstm_model.pt")

            with open(cfg_path, "r", encoding="utf8") as f:
                cfg = json.load(f)

            with open(vocab_path, "r", encoding="utf8") as f:
                vocab = json.load(f)

            # создаём модель
            model = LSTMClassifier(
                vocab_size=cfg["vocab_size"],
                embedding_dim=cfg["embedding_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"]
            )

            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            return {"type": "lstm", "model": model, "vocab": vocab}, None

        # === BERT ===
        elif "BERT" in model_name:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                repo_id,
                output_attentions=True
            )
            model.eval()

            return {"type": "bert", "model": model, "tokenizer": tokenizer}, None

        else:
            return None, "Неизвестная модель"

    except Exception as e:
        return None, str(e)


# === Сайдбар ===
with st.sidebar:
    st.header("⚙️ Настройки модели")

    model_choice = st.selectbox(
        "Выберите модель",
        list(MODEL_PATHS.keys()),
        help="Baseline: TF-IDF + LogReg | LSTM | BERT"
    )

    model_path = st.text_input(
        "Путь к модели",
        value=MODEL_PATHS[model_choice]
    )

    load_model_btn = st.button("📥 Загрузить модель", type="primary", use_container_width=True)

    if load_model_btn and model_path:
        with st.spinner(f"Загрузка {model_choice}..."):
            model_data, error = load_model_cached(model_choice, model_path)
            if model_data:
                st.session_state.model_loaded = True
                st.session_state.current_model = model_choice
                st.session_state.model_path = model_path
                st.session_state.model_data = model_data
                st.success(f"✅ {model_choice} загружена!")
            else:
                st.error(f"❌ Ошибка: {error}")

    if st.session_state.model_loaded:
        st.success(f"Активна: {st.session_state.current_model}")
        if st.button("🔄 Сменить модель"):
            st.session_state.model_loaded = False
            st.session_state.last_result = None
            st.rerun()

    st.divider()
    threshold = st.slider("Порог уверенности", 0.5, 0.99, 0.7, 0.05)

    show_attention = st.checkbox(
        "Показывать Attention (BERT)",
        value=True,
        disabled="BERT" not in model_choice
    )

    st.divider()
    st.subheader("📊 Статистика")
    stats = st.session_state.stats
    st.metric("Всего отзывов", stats["total"])

    if stats["total"] > 0:
        df_stat = pd.DataFrame({
            "Тональность": ['Негативный', 'Позитивный'],
            "Количество": [
                stats["by_sentiment"]['Негативный'],
                stats["by_sentiment"]['Позитивный']
            ]
        })
        st.bar_chart(df_stat.set_index("Тональность"))


# === Основной интерфейс ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Ввод отзыва")
    user_text = st.text_area(
        "Текст отзыва",
        value=st.session_state.user_text,
        height=150,
        placeholder="Например: Врач был внимательным..."
    )

    with st.expander("📋 Примеры"):
        examples = {
            "Позитивный": "Отличный врач! Все объяснил, лечение помогло.",
            "Негативный": "Очень долго ждал приема, грубый персонал."
        }
        for sentiment, text in examples.items():
            if st.button(f"🎯 {sentiment}", key=f"ex_{sentiment}"):
                st.session_state.user_text = text
                st.rerun()

with col2:
    st.subheader("⚡ Действия")
    classify_btn = st.button(
        "🔍 Классифицировать",
        type="primary",
        use_container_width=True,
        disabled=not user_text
    )
    if st.button("🧹 Очистить", use_container_width=True):
        st.session_state.user_text = ""
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.info("🎯 Бинарная классификация (Негативный / Позитивный)\nF1-macro")


# === Инференс ===
if classify_btn and user_text:
    with st.spinner("🤖 Анализируем..."):
        start = time.time()

        try:
            model_data = st.session_state.model_data
            model_type = model_data["type"]

            # ==== LogReg ====
            if model_type == "logreg":
                clf = model_data["model"]
                probs = clf.predict_proba([user_text])[0]

            # ==== LSTM ====
            elif model_type == "lstm":
                model = model_data["model"]
                vocab = model_data["vocab"]

                token_ids = [
                    vocab.get(tok.lower(), vocab.get("<unk>", 0))
                    for tok in user_text.split()
                ]
                if not token_ids:
                    token_ids = [vocab.get("<pad>", 0)]

                ids = torch.tensor(token_ids).unsqueeze(0)

                with torch.no_grad():
                    logits = model(ids)
                    probs = torch.softmax(logits, dim=1).squeeze().tolist()

            # ==== BERT ====
            elif model_type == "bert":
                model = model_data["model"]
                tokenizer = model_data["tokenizer"]

                inputs = tokenizer(
                    user_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_attentions=True,
                        return_dict=True
                    )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze().tolist()

                st.session_state.attentions = outputs.attentions
                st.session_state.tokens = inputs

            else:
                raise RuntimeError("Неизвестный тип модели")

            sorted_idx = sorted(range(2), key=lambda i: probs[i], reverse=True)

            result = {
                "text": user_text,
                "model": st.session_state.current_model,
                "labels": [SENTIMENT_LABELS[i] for i in sorted_idx],
                "scores": [probs[i] for i in sorted_idx],
                "predicted": SENTIMENT_LABELS[sorted_idx[0]],
                "confidence": probs[sorted_idx[0]],
                "time": time.time() - start,
                "uncertain": probs[sorted_idx[0]] < threshold
            }

            st.session_state.last_result = result

            # === обновление статистики
            stats = st.session_state.stats
            stats["total"] += 1
            stats["by_sentiment"][result["predicted"]] += 1
            stats["by_model"].setdefault(result["model"], 0)
            stats["by_model"][result["model"]] += 1

        except Exception as e:
            st.error(f"Ошибка: {e}")

# === HTML-подсветка токенов по attention ===
def attention_to_html(tokens, attn_vec):
    """
    Возвращает HTML-строку с подсветкой токенов по силе attention.
    Используется в Streamlit через markdown(..., unsafe_allow_html=True)
    """

    # нормируем значения внимания к [0,1]
    attn_norm = (attn_vec - attn_vec.min()) / (attn_vec.max() - attn_vec.min() + 1e-8)

    html_tokens = []

    for tok, score in zip(tokens, attn_norm):
        alpha = float(score)

        # от белого (низкое внимание) к оранжевому (высокое)
        color = f"rgba(255, 165, 0, {alpha:.2f})"

        # убираем служебный символ BERT
        tok_clean = tok.replace("▁", "")
        if tok_clean == "":
            tok_clean = tok  # fallback

        span = f"<span style='background-color:{color}; padding:2px; margin:1px; border-radius:3px;'>{tok_clean}</span>"
        html_tokens.append(span)

    return "<div>" + " ".join(html_tokens) + "</div>"

# === Результаты ===
if st.session_state.last_result:
    res = st.session_state.last_result

    st.divider()
    st.subheader("📊 Результат")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Тональность", res["predicted"])
    colB.metric("Уверенность", f"{res['confidence']:.1%}")
    colC.metric("Модель", res["model"])
    colD.metric("Время", f"{res['time']*1000:.0f} мс")

    st.markdown("### 📈 Вероятности")
    df = pd.DataFrame({
        "Класс": res["labels"],
        "Score": res["scores"]
    })
    st.bar_chart(df.set_index("Класс"))

    if res["model"].startswith("BERT") and show_attention:
        attentions = st.session_state.attentions
        tokens_raw = st.session_state.tokens["input_ids"][0]

        if not attentions or len(attentions) == 0:
            st.warning("❗ Эта модель не возвращает attention. Проверь config.json.")
        else:
            last_layer = attentions[-1]
            cls_attention = last_layer[0].mean(dim=0)[0]
            tokenizer = st.session_state.model_data["tokenizer"]
            tokens = tokenizer.convert_ids_to_tokens(tokens_raw)

            html = attention_to_html(tokens, cls_attention.cpu().numpy())
            st.markdown("### 🔍 Важность токенов (Attention)", unsafe_allow_html=False)
            st.markdown(html, unsafe_allow_html=True)

 

# === Сравнение моделей ===
st.divider()
st.subheader("📊 Сравнение моделей (F1-macro)")

metrics_data = pd.DataFrame({
    "Модель": list(MODEL_PATHS.keys()),
    "F1-macro": [0.94, 0.93, 0.94],
    "Время (minutes)": [0.36, 2.06, 11.4]
})

col_chart, col_table = st.columns(2)

with col_chart:
    st.bar_chart(metrics_data.set_index("Модель")[["F1-macro"]])

with col_table:
    st.dataframe(
        metrics_data,
        hide_index=True,
        use_container_width=True
    )

st.divider()
st.caption("🏥 Бинарная классификация отзывов | NLP demo")



import joblib
model = joblib.load("./models/reviews/baseline_model.pkl")
print(model.predict(["Очень долго ждал приема, грубый персонал."]))
print(model.predict_proba(["Очень долго ждал приема, грубый персонал."]))
