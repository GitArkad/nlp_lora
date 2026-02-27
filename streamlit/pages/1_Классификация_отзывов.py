import streamlit as st
import torch
import time
import pandas as pd
from utils import page_header

# === 🔥 ИНИЦИАЛИЗАЦИЯ SESSION_STATE (САМОЕ НАЧАЛО!) ===
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
    "Baseline (LogReg)": "./models/baseline_logreg.pkl",
    "LSTM": "./models/lstm_sentiment.pt",
    "BERT (rubert-tiny2)": "./models/rubert-tiny2-sentiment"
}

st.set_page_config(page_title="Отзывы", page_icon="🏥", layout="wide")
page_header("Классификация отзывов на поликлиники", "🏥")

# === Загрузка модели (кэш) ===
@st.cache_resource
def load_model_cached(model_name, model_path):
    """Кэшированная загрузка модели"""
    try:
        if "LogReg" in model_name:
            # import joblib
            # model = joblib.load(model_path)
            # vectorizer = joblib.load(model_path.replace('.pkl', '_vectorizer.pkl'))
            return {"type": "logreg"}, None
        elif "LSTM" in model_name:
            # model = torch.load(model_path, map_location='cpu')
            return {"type": "lstm"}, None
        elif "BERT" in model_name:
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # tokenizer = AutoTokenizer.from_pretrained(model_path)
            # model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            return {"type": "bert"}, None
        return None, None
    except Exception as e:
        return None, str(e)

# === Сайдбар ===
with st.sidebar:
    st.header("⚙️ Настройки модели")
    
    model_choice = st.selectbox(
        "Выберите модель",
        list(MODEL_PATHS.keys()),
        help="Baseline: TF-IDF + LogReg | LSTM: рекуррентная сеть | BERT: трансформер"
    )
    
    model_path = st.text_input(
        "Путь к модели",
        value=MODEL_PATHS[model_choice],
        help="Путь к файлу модели"
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
                st.info("💡 Используется демо-режим")
    
    if st.session_state.model_loaded:
        st.success(f"✅ Активна: {st.session_state.current_model}")
        if st.button("🔄 Сменить модель"):
            st.session_state.model_loaded = False
            st.session_state.last_result = None
            st.rerun()
    
    st.divider()
    
    threshold = st.slider(
        "Порог уверенности",
        min_value=0.5, max_value=0.99, value=0.7, step=0.05
    )
    
    show_attention = st.checkbox(
        "Показывать Attention (BERT)",
        value=True,
        disabled="BERT" not in model_choice
    )
    
    st.divider()
    
    # Статистика — безопасное чтение
    st.subheader("📊 Статистика сессии")
    stats = st.session_state.get("stats", {})
    total = stats.get("total", 0)
    by_sentiment = stats.get("by_sentiment", {'Негативный': 0, 'Позитивный': 0})
    
    st.metric("Всего отзывов", total)
    
    if any(by_sentiment.values()):
        stats_df = pd.DataFrame({
            "Тональность": ['Негативный', 'Позитивный'],
            "Количество": [by_sentiment.get('Негативный', 0), by_sentiment.get('Позитивный', 0)]
        })
        st.bar_chart(stats_df.set_index("Тональность"), use_container_width=True)

# === Основной интерфейс ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Ввод отзыва")
    user_text = st.text_area(
        "Текст отзыва",
        value=st.session_state.user_text,
        height=150,
        placeholder="Например: Врач был внимательным, но долго ждал...",
        key="text_input"
    )
    
    with st.expander("📋 Примеры"):
        examples = {
            "Позитивный": "Отличный врач! Всё объяснил, лечение помогло. Спасибо!",
            "Негативный": "Ужасное отношение, грубость, долго ждал. Не рекомендую."
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
    st.info("🎯 **Задача:** Бинарная классификация\n\n📊 **Классы:** `Негативный` / `Позитивный`\n\n⚡ **Метрика:** F1-macro")

# === Инференс ===
if classify_btn and user_text:
    with st.spinner("🤖 Анализируем..."):
        start_time = time.time()
        
        try:
            # TODO: Реальный инференс
            if st.session_state.model_loaded:
                # Вставьте код предсказания здесь
                pass
            
            # Заглушка (демо)
            import random
            pos_score = random.uniform(0.5, 0.99)
            neg_score = 1 - pos_score
            if random.random() > 0.5:
                pos_score, neg_score = neg_score, pos_score
            
            probs = [neg_score, pos_score]
            sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            
            inference_time = time.time() - start_time
            
            result = {
                "text": user_text,
                "model": model_choice if st.session_state.model_loaded else "Demo",
                "labels": [SENTIMENT_LABELS[i] for i in sorted_idx],
                "scores": [probs[i] for i in sorted_idx],
                "predicted": SENTIMENT_LABELS[sorted_idx[0]],
                "confidence": probs[sorted_idx[0]],
                "time": inference_time,
                "uncertain": probs[sorted_idx[0]] < threshold
            }
            
            st.session_state.last_result = result
            
            # Обновление статистики
            st.session_state.stats["total"] += 1
            st.session_state.stats["by_sentiment"][result["predicted"]] += 1
            model_name = result["model"]
            if model_name not in st.session_state.stats["by_model"]:
                st.session_state.stats["by_model"][model_name] = 0
            st.session_state.stats["by_model"][model_name] += 1
            
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

# === Результаты ===
if st.session_state.last_result:
    res = st.session_state.last_result
    
    st.divider()
    st.subheader("📊 Результат")
    
    if res["predicted"] == "Позитивный":
        color, emoji = "🟢", "😊"
    else:
        color, emoji = "🔴", "😞"
    
    if res["uncertain"]:
        st.warning(f"⚠️ Низкая уверенность: {res['confidence']:.1%} < порог {threshold:.0%}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Тональность", f"{color} {emoji} {res['predicted']}")
    col2.metric("Уверенность", f"{res['confidence']:.1%}")
    col3.metric("Модель", res["model"].split()[0])
    col4.metric("Время", f"{res['time']*1000:.0f} мс")
    
    st.markdown("### 📈 Вероятности")
    
    df = pd.DataFrame({
        "Класс": res["labels"],
        "Вероятность": [f"{s:.1%}" for s in res["scores"]],
        "Score": res["scores"]
    })
    
    colors = {"Позитивный": "#4CAF50", "Негативный": "#F44336"}
    st.bar_chart(
        df.set_index("Класс")[["Score"]],
        use_container_width=True,
        color=[colors.get(l, "#90A4AE") for l in df["Класс"]]
    )
    
    with st.expander("📋 Детали"):
        st.dataframe(
            df[["Класс", "Вероятность"]],
            use_container_width=True,
            hide_index=True
        )
        st.json({
            "predicted_label": res["predicted"],
            "predicted_id": LABEL_MAPPING[res["predicted"]],
            "scores": dict(zip(res["labels"], res["scores"]))
        })
    
    if "BERT" in res["model"] and show_attention:
        st.markdown("### 🔍 Attention (BERT)")
        st.info("💡 Токены, на которые модель обращала внимание")
        st.markdown(f"Текст: `{res['text'][:100]}...`")
        st.progress(res["confidence"])

# === Сравнение моделей ===
st.divider()
st.subheader("📊 Сравнение моделей (F1-macro)")

metrics_data = pd.DataFrame({
    "Модель": list(MODEL_PATHS.keys()),
    "F1-macro": [0.82, 0.89, 0.94],
    "Время (мс)": [15, 45, 120]
})

col_chart, col_table = st.columns(2)

with col_chart:
    st.bar_chart(
        metrics_data.set_index("Модель")[["F1-macro"]],
        use_container_width=True,
        color="#4CAF50"
    )

with col_table:
    st.dataframe(
        metrics_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "F1-macro": st.column_config.ProgressColumn("F1-macro", min_value=0, max_value=1, format="%.2f"),
            "Время (мс)": st.column_config.NumberColumn("Время (мс)")
        }
    )

# === Футер ===
st.divider()
st.caption(f"🏥 Бинарная классификация: {', '.join(SENTIMENT_LABELS)} | Заглушка моделей")
# подсказки по замене То есть найти в коде и исрпавить c учетом заметок

#удалить
# В функции load_model_cached:
# if "LogReg" in model_name:
#     # import joblib
#     # model = joblib.load(model_path)
#     return {"type": "logreg"}, None
# elif "LSTM" in model_name:
#     return {"type": "lstm"}, None
# elif "BERT" in model_name:
#     return {"type": "bert"}, None




#@st.cache_resource
# def load_model_cached(model_name, model_path):!!!!!!!!!!!!заменить пути и
#     """Кэшированная загрузка модели"""
#     try:
#         if "LogReg" in model_name:
#             import joblib
#             from sklearn.feature_extraction.text import TfidfVectorizer
            
#             # Загрузка векторайзера и модели
#             vectorizer = joblib.load(model_path.replace('.pkl', '_vectorizer.pkl'))
#             model = joblib.load(model_path)
            
#             return {"type": "logreg", "model": model, "vectorizer": vectorizer}, None
            
#         elif "LSTM" in model_name:
#             # Для LSTM модели
#             model = torch.load(model_path, map_location='cpu')
#             model.eval()
            
#             return {"type": "lstm", "model": model}, None
            
#         elif "BERT" in model_name:
#             from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
#             tokenizer = AutoTokenizer.from_pretrained(model_path)
#             model = AutoModelForSequenceClassification.from_pretrained(
#                 model_path,
#                 num_labels=2,
#                 id2label=ID_TO_LABEL,
#                 label2id=LABEL_MAPPING
#             )
#             model.eval()
            
#             if torch.cuda.is_available():
#                 model = model.to('cuda')
            
#             return {"type": "bert", "model": model, "tokenizer": tokenizer}, None
            
#         return None, "Неизвестный тип модели"
        
#     except Exception as e:
#         return None, str(e)




## --- УДАЛИТЬ ЭТОТ БЛОК ПОЛНОСТЬЮ ---
# Заглушка (демо)
# import random
# pos_score = random.uniform(0.5, 0.99)
# neg_score = 1 - pos_score
# if random.random() > 0.5:
#     pos_score, neg_score = neg_score, pos_score

# probs = [neg_score, pos_score]
# sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
# # -----------------------------------