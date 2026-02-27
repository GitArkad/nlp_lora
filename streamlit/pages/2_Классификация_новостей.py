import streamlit as st
import torch
import pandas as pd
import os
import joblib
import sys
from datetime import datetime

# === Конфигурация страницы ===
st.set_page_config(
    page_title="Классификация новостей", 
    page_icon="📰",
    layout="centered"
)

# === Заголовок ===
st.title("📰 Классификация новостей Telegram")
st.markdown("---")

# === Категории ===
CATEGORIES = ["мода", "технологии", "финансы", "крипта", "спорт"]

# === Простая инициализация session state ===
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# === Создаем точную копию модели из чекпоинта ===
class ExactLSTMModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # Параметры из чекпоинта
        self.embedding_dim = 256
        self.lstm_hidden = 128  # Из размерностей: weight_hh_l0 имеет [512, 128] -> hidden_size=128
        self.num_layers = 2
        self.bidirectional = True
        
        # Embedding слой
        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        
        # LSTM слой (bidirectional)
        self.lstm = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=self.bidirectional
        )
        
        # Self-attention (из размерностей: in_proj_weight [768, 256])
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=self.lstm_hidden * 2,  # 256
            num_heads=4,  # 768/3/256 = 4 heads
            dropout=0.1,
            batch_first=True
        )
        
        # Классификатор (из ключей clf.0, clf.1, clf.4, clf.5, clf.8)
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(256, 128),  # clf.0
            torch.nn.ReLU(),              # clf.1
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),     # clf.4
            torch.nn.ReLU(),              # clf.5
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 5)         # clf.8
        )
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, 256]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, 256]
        
        # Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)  # [batch, seq_len, 256]
        
        # Используем среднее по всем токенам
        pooled = torch.mean(attn_out, dim=1)  # [batch, 256]
        
        # Классификация
        output = self.clf(pooled)  # [batch, 5]
        
        return output

# Добавляем класс в __main__ для возможности загрузки
sys.modules['__main__'].ExactLSTMModel = ExactLSTMModel

# === Функции загрузки моделей ===

def load_logreg_model():
    """Загрузка логистической регрессии"""
    try:
        model_path = "models/log_reg_tg/model_logreg.pkl"
        vec_path = "models/log_reg_tg/tfidf_vectorizer.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            return None
            
        model = joblib.load(model_path)
        
        if not os.path.exists(vec_path):
            st.error(f"Файл векторизатора не найден: {vec_path}")
            return None
            
        vectorizer = joblib.load(vec_path)
        return {"type": "logreg", "model": model, "vectorizer": vectorizer}
    except Exception as e:
        st.error(f"Ошибка загрузки LogReg: {e}")
        return None

def load_lstm_model():
    """Загрузка LSTM модели с точной архитектурой"""
    try:
        model_path = "models/lstm-tg/model_lstm_weights.pt"
        vocab_path = "models/lstm-tg/vocab_to_int_lstm.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"Файл модели не найден: {model_path}")
            return None
        
        if not os.path.exists(vocab_path):
            st.error(f"Файл словаря не найден: {vocab_path}")
            return None
            
        # Загружаем словарь
        vocab = joblib.load(vocab_path)
        vocab_size = len(vocab) + 1
        
        st.info(f"Размер словаря: {vocab_size}")
        
        # Загружаем чекпоинт
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Создаем модель с точной архитектурой
        model = ExactLSTMModel(vocab_size)
        
        # Загружаем веса с strict=False
        model.load_state_dict(checkpoint, strict=False)
        st.info("Веса загружены с пропуском несоответствий")
        
        model.eval()
        
        return {"type": "lstm", "model": model, "vocab": vocab}
        
    except Exception as e:
        st.error(f"Ошибка загрузки LSTM: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# === Боковая панель ===
with st.sidebar:
    st.header("⚙️ Выбор модели")
    
    st.info("""
    **Доступные модели:**
    - ✅ LogReg (работает)
    - LSTM (точная архитектура)
    """)
    
    model_choice = st.selectbox(
        "Модель:",
        ["LogReg", "LSTM"]
    )
    
    if st.button("📥 Загрузить модель", use_container_width=True, type="primary"):
        with st.spinner(f"Загрузка {model_choice}..."):
            
            if model_choice == "LogReg":
                model_data = load_logreg_model()
            else:  # LSTM
                model_data = load_lstm_model()
            
            if model_data:
                st.session_state.model = model_data
                st.session_state.model_name = model_choice
                st.success(f"✅ {model_choice} загружена!")
                st.rerun()
    
    if st.session_state.model:
        st.success(f"✅ Активна: **{st.session_state.model_name}**")
        if st.button("🔄 Сбросить"):
            st.session_state.model = None
            st.session_state.model_name = None
            st.rerun()

# === Основной интерфейс ===
st.subheader("📝 Введите текст для классификации")

# Поле для ввода текста
text = st.text_area(
    "Текст новости:",
    height=150,
    placeholder="Введите текст новости из Telegram...",
    key="input_text"
)

# Кнопки
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🔍 Классифицировать", use_container_width=True, type="primary", 
                           disabled=st.session_state.model is None)
with col2:
    if st.button("🗑️ Очистить", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()

# Предупреждение если модель не загружена
if st.session_state.model is None:
    st.warning("⚠️ Сначала загрузите модель в боковой панели!")

# === Предсказание ===
if predict_btn and text and st.session_state.model:
    with st.spinner("Анализ..."):
        try:
            model_data = st.session_state.model
            model_type = model_data["type"]
            
            if model_type == "logreg":
                # LogReg предсказание
                vectorizer = model_data["vectorizer"]
                text_vectorized = vectorizer.transform([text])
                probs = model_data["model"].predict_proba(text_vectorized)[0]
                
            else:  # LSTM
                vocab = model_data["vocab"]
                
                # Токенизация текста
                words = text.lower().split()
                tokens = []
                for word in words:
                    if word in vocab:
                        tokens.append(vocab[word])
                    else:
                        tokens.append(0)  # 0 для неизвестных слов
                
                # Паддинг до фиксированной длины
                max_len = 100
                if len(tokens) < max_len:
                    tokens = tokens + [0] * (max_len - len(tokens))
                else:
                    tokens = tokens[:max_len]
                
                # Предсказание
                input_tensor = torch.tensor([tokens], dtype=torch.long)
                with torch.no_grad():
                    output = model_data["model"](input_tensor)
                    probs = torch.softmax(output, dim=-1)[0].numpy()
            
            # Результат
            pred_idx = probs.argmax()
            confidence = probs[pred_idx]
            
            st.session_state.last_result = {
                "category": CATEGORIES[pred_idx],
                "confidence": confidence,
                "probabilities": dict(zip(CATEGORIES, probs)),
                "model": st.session_state.model_name
            }
            
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

# === Отображение результатов ===
if st.session_state.last_result:
    res = st.session_state.last_result
    
    st.markdown("---")
    st.subheader("📊 Результат")
    
    st.caption(f"Модель: **{res['model']}**")
    
    conf = res["confidence"]
    if conf >= 0.7:
        color = "🟢"
    elif conf >= 0.5:
        color = "🟡"
    else:
        color = "🔴"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Категория", f"{color} {res['category'].upper()}")
    with col2:
        st.metric("Уверенность", f"{conf:.1%}")
    
    # График вероятностей
    st.markdown("### 📈 Вероятности по категориям")
    prob_df = pd.DataFrame({
        'Категория': list(res['probabilities'].keys()),
        'Вероятность': list(res['probabilities'].values())
    })
    st.bar_chart(prob_df.set_index('Категория'))

st.markdown("---")
st.caption("📰 Классификация новостей Telegram")