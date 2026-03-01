import streamlit as st
import torch
import torch.nn as nn
import transformers
import joblib
import pickle
import json
import os
import time
from datetime import datetime
import re
from pymorphy3 import MorphAnalyzer
from pathlib import Path

# Инициализация морфологического анализатора (один раз при старте)
morph = MorphAnalyzer()

# Стоп-слова (идентичны обучению)
STOP_WORDS = {
    'который', 'свой', 'быть', 'этот', 'очень', 'самый', 'наш', 'ваш', 'их',
    'стать', 'мочь', 'хотеть', 'сделать', 'иметь', 'являться', 'получить', 'дать',
    'пользователь', 'клиент', 'сервис', 'платформа', 'приложение', 'система',
    'компания', 'проект', 'работа', 'новый', 'возможность', 'использование',
    'позволять', 'данный', 'нужно', 'необходимый', 'рамка', 'сообщать', 'пик',
    'читать', 'подписаться', 'канал', 'пост', 'новость', 'подробность', 'ссылка'
}

def clean_lstm_improved(text):
    """Очистка текста для LSTM — 1:1 как при обучении."""
    if not isinstance(text, str):
        return ""
    
    # 1. Удаление ссылок и упоминаний
    text = re.sub(r'https?://\S+|t\.me/\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    
    # 2. Чистка пунктуации, сохранение хэштегов
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^\w\s#]', ' ', text)
    text = text.lower()
    
    tokens = []
    for word in text.split():
        # Хэштеги сохраняем как есть
        if word.startswith('#'):
            tokens.append(word)
            continue
        
        # Лемматизация
        p = morph.parse(word)[0]
        lemma = p.normal_form
        pos = p.tag.POS
        
        # Фильтрация
        if lemma in STOP_WORDS:
            continue
        if pos in {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}:
            continue
        if len(lemma) < 3 and not word.isascii():
            continue
        if lemma not in STOP_WORDS:
            tokens.append(lemma)
    
    return " ".join(tokens)

st.set_page_config(
    page_title="Классификация новостей",
    page_icon="📰",
    layout="wide"
)

# === ИНИЦИАЛИЗАЦИЯ SESSION STATE ===
if 'news_model_loaded' not in st.session_state:
    st.session_state.news_model_loaded = None
if 'news_model_data' not in st.session_state:
    st.session_state.news_model_data = {}
if 'news_prediction_result' not in st.session_state:
    st.session_state.news_prediction_result = None
if 'news_input_text' not in st.session_state:
    st.session_state.news_input_text = ""

# === КОНСТАНТЫ ===
MODEL_METRICS = {
    "LogReg": {
        "accuracy": 0.9332,
        "f1_macro": 0.9342  ,
        "dataset_size": 2900,
        "confusion_matrix_url": "models/log_reg_tg/cm_logreg.png"
    },
    "LSTM": {
        "accuracy": 0.8524,
        "f1_macro": 0.8498,
        "dataset_size": 2900,
        "confusion_matrix_url": "models/lstm_tg/cm_lstm.png"
    },
    "RuBERT": {
        "accuracy": 0.9561,
        "f1_macro": 0.9568,
        "dataset_size": 2900,
        "confusion_matrix_url": "models/rubert_tiny_tg/cm_rubert.png"
    }
}

MODEL_PATHS = {
    "LogReg": "models/log_reg_tg",
    "LSTM": "models/lstm_tg",
    "RuBERT": "models/rubert_tiny_tg"
}

# git add .gitattributes

# === КЛАССЫ МОДЕЛЕЙ ===
class LSTMClassifier0(nn.Module):
    def __init__(self, rnn_conf, num_classes=5):
        """
        Классификатор тем на основе LSTM.
        rnn_conf: dict с ключами vocab_size, embedding_dim, hidden_size, etc.
        """
        super().__init__()
        
        # 1. Параметры (доступ через ['key'] для dict)
        self.embedding_dim = rnn_conf['embedding_dim']
        self.hidden_size = rnn_conf['hidden_size']
        self.bidirectional = rnn_conf['bidirectional']
        self.n_layers = rnn_conf['n_layers']
        self.bidirect_factor = 2 if self.bidirectional else 1
        lstm_out_dim = self.hidden_size * self.bidirect_factor
        
        # 2. Embedding
        self.embedding = nn.Embedding(rnn_conf['vocab_size'], self.embedding_dim, padding_idx=0)
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=0.5 if self.n_layers > 1 else 0.0
        )
        
        # 4. Классификатор (имя 'classifier' и вход lstm_out_dim * 2 — как в весах!)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)  # [B, L, E]
        lstm_out, _ = self.lstm(embedded)  # [B, L, H*2]
        
        # Avg + Max Pooling
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        combined = torch.cat((avg_pool, max_pool), dim=1)  # [B, H*2*2]
        
        return self.classifier(combined)

class MyBERTUnFreeze(nn.Module):
    def __init__(self, num_labels=5, model_name="cointegrated/rubert-tiny2"):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size 
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_out.pooler_output
        logits = self.linear(pooled_output)
        return logits

# === ФУНКЦИИ ЗАГРУЗКИ ===
@st.cache_resource
def load_logreg():
    path = MODEL_PATHS["LogReg"]
    model = joblib.load(os.path.join(path, 'model_logreg.pkl'))
    vectorizer = joblib.load(os.path.join(path, 'tfidf_vectorizer.pkl'))
    class_mapping = joblib.load(os.path.join(path, 'class_mapping.pkl'))
    # with open(os.path.join(path, 'class_mapping.json'), 'r', encoding='utf-8') as f:
    #     class_mapping = json.load(f)
    return model, vectorizer, class_mapping

@st.cache_resource
def load_lstm():
    path = MODEL_PATHS["LSTM"]
    with open(os.path.join(path, 'rnn_config.json'), 'r', encoding='utf-8') as f:
        rnn_conf = json.load(f)
    with open(os.path.join(path, 'preprocessing_config.json'), 'r', encoding='utf-8') as f:
        prep_conf = json.load(f)
    with open(os.path.join(path, 'vocab_to_int.pkl'), 'rb') as f:
        vocab_to_int = pickle.load(f)
    with open(os.path.join(path, 'class_mapping.json'), 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    
    model = LSTMClassifier0(rnn_conf)
    model.load_state_dict(torch.load(os.path.join(path, 'model_lstm_weights.pt'), 
                                      map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model, vocab_to_int, prep_conf, class_mapping

@st.cache_resource
def load_rubert():
    path = MODEL_PATHS["RuBERT"]
    with open(os.path.join(path, 'model_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(os.path.join(path, 'class_mapping.json'), 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    model = MyBERTUnFreeze(num_labels=config['num_labels'], model_name=config['model_name'])
    model.load_state_dict(torch.load(os.path.join(path, 'model_tg_weights.pt'), 
                                      map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model, tokenizer, class_mapping

# === ЗАГОЛОВОК ===
st.title("📰 Классификация новостей Telegram")
st.markdown("---")

# === ВЫБОР МОДЕЛИ ===
col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox(
        "Выберите модель:",
        options=["LogReg", "LSTM", "RuBERT"],
        format_func=lambda x: f"{'📊' if x == 'LogReg' else '🧠' if x == 'LSTM' else '🦀'} {x}",
        key="news_model_selector"
    )

# === ЗАГРУЗКА МОДЕЛИ ПРИ ИЗМЕНЕНИИ ВЫБОРА ===
if model_choice != st.session_state.news_model_loaded:
    with st.spinner(f"Загрузка модели {model_choice}..."):
        if model_choice == "LogReg":
            model, vectorizer, class_mapping = load_logreg()
            st.session_state.news_model_data = {
                'model': model,
                'vectorizer': vectorizer,
                'class_mapping': class_mapping,
                'type': 'logreg'
            }
        elif model_choice == "LSTM":
            model, vocab_to_int, prep_conf, class_mapping = load_lstm()
            st.session_state.news_model_data = {
                'model': model,
                'vocab_to_int': vocab_to_int,
                'prep_conf': prep_conf,
                'class_mapping': class_mapping,
                'type': 'lstm'
            }
        else:
            model, tokenizer, class_mapping = load_rubert()
            st.session_state.news_model_data = {
                'model': model,
                'tokenizer': tokenizer,
                'class_mapping': class_mapping,
                'type': 'rubert'
            }
        st.session_state.news_model_loaded = model_choice
        st.session_state.news_prediction_result = None
    st.success(f"✅ Модель {model_choice} загружена!")

# === ОСНОВНОЙ ЛЕЙАУТ: СЛЕВА ВВОД, СПРАВА ИНФО ===
main_col1, main_col2 = st.columns([1.5, 1], gap="large")

with main_col1:
    st.markdown("### 📝 Введите текст новости")
    user_text = st.text_area(
        "Текст новости",
        value=st.session_state.news_input_text,
        height=250,
        placeholder="Вставьте текст новости здесь...",
        key="news_text_input"
    )
    
    st.session_state.news_input_text = user_text
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        predict_button = st.button("🔮 Предсказать категорию", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Очистить"):
            st.session_state.news_input_text = ""
            st.session_state.news_prediction_result = None
            st.rerun()
    
    # === ПРЕДСКАЗАНИЕ ===
    # ✅ ВАЖНО: if и elif на одном уровне отступов (с начала строки)!
    if predict_button and user_text.strip():
        start_time = time.time()
        
        try:
            model_data = st.session_state.news_model_data
            
            if model_data['type'] == 'logreg':
                input_data = model_data['vectorizer'].transform([user_text])
                probs = model_data['model'].predict_proba(input_data)[0]
            elif model_data['type'] == 'lstm':
                # 1. Очистка текста (1:1 как при обучении)
                cleaned = clean_lstm_improved(user_text)
                
                # 2. Токенизация + индексация (пропускаем неизвестные слова)
                tokens = cleaned.split()
                indices = [model_data['vocab_to_int'][token] 
                          for token in tokens 
                          if token in model_data['vocab_to_int']]
                
                # 3. Левый паддинг (как при обучении!)
                max_len = model_data['prep_conf']['max_len']
                padding_value = model_data['prep_conf']['padding_value']
                if len(indices) < max_len:
                    indices = [padding_value] * (max_len - len(indices)) + indices  # ← Левый!
                else:
                    indices = indices[:max_len]
                
                # 4. Инференс
                input_tensor = torch.tensor([indices], dtype=torch.long)
                with torch.no_grad():
                    output = model_data['model'](input_tensor)
                    probs = torch.softmax(output, dim=1)[0].numpy()
            else:
                inputs = model_data['tokenizer'](user_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    logits = model_data['model'](**inputs)
                    probs = torch.softmax(logits, dim=1)[0].numpy()
            
            inference_time = (time.time() - start_time) * 1000
            top_indices = probs.argsort()[::-1][:2]
            top1_idx, top2_idx = top_indices[0], top_indices[1]
            top1_conf, top2_conf = probs[top1_idx], probs[top2_idx]
            
            def get_label(idx):
                return model_data['class_mapping'].get(str(idx), model_data['class_mapping'].get(idx, f"Класс {idx}"))
            
            st.session_state.news_prediction_result = {
                'label1': get_label(top1_idx),
                'conf1': top1_conf,
                'label2': get_label(top2_idx),
                'conf2': top2_conf,
                'time': inference_time,
                'all_probs': probs,
                'class_mapping': model_data['class_mapping']
            }
            
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {str(e)}")
            st.session_state.news_prediction_result = None

    # ✅ elif на том же уровне, что и if (с начала строки!)
    if predict_button and not user_text.strip():
        st.warning("⚠️ Введите текст новости для предсказания")
    
    # === ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ===
    if st.session_state.news_prediction_result:
        result = st.session_state.news_prediction_result
        
        st.markdown("---")
        st.markdown("### 🎯 Результаты предсказания")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.success(f"### 🏆 {result['label1']}")
            st.progress(float(result['conf1']))
            st.write(f"**Уверенность:** {result['conf1']:.2%}")
        
        # with res_col2:
        #     st.info(f"### 🥈 {result['label2']}")
        #     st.progress(float(result['conf2']))
        #     st.write(f"**Уверенность:** {result['conf2']:.2%}")
        
        st.markdown("---")
        
        time_col1, time_col2, time_col3 = st.columns(3)
        with time_col1:
            st.metric("⏱ Время предсказания", f"{result['time']:.1f} мс")
        with time_col2:
            st.metric("📊 Модель", model_choice)
        with time_col3:
            st.metric("🔢 Всего классов", len(result['class_mapping']))
        
        with st.expander("📊 Все вероятности по классам"):
            for idx in range(len(result['all_probs'])):
                label = result['class_mapping'].get(str(idx), result['class_mapping'].get(idx, f"Класс {idx}"))
                prob = result['all_probs'][idx]
                st.write(f"{label}: {prob:.2%}")

with main_col2:
    st.markdown("### 📊 Информация о модели")
    metrics = MODEL_METRICS[model_choice]
    
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("F1-Macro", f"{metrics['f1_macro']:.2%}")
    st.metric("Размер датасета", f"{metrics['dataset_size']:,}")
    st.metric("Время инференса", "~10-200 мс")
    
    st.markdown("---")
    
    if metrics['confusion_matrix_url']:
        cm_path = Path(metrics['confusion_matrix_url'])
        
        if cm_path.exists():
            st.image(str(cm_path), caption=f"Confusion Matrix - {model_choice}", width='stretch')
        else:
            st.warning(f"⚠️ Файл не найден: `{cm_path}`")
        
        # st.markdown(f"[📈 Открыть Confusion Matrix]({metrics['confusion_matrix_url']})")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Описание модели")
    if model_choice == "LogReg":
        st.info("""
        **Логистическая регрессия + TF-IDF**
        
        - Быстрая и легковесная
        - Хорошо работает на простых текстах
        - Интерпретируемые результаты
        """)
    elif model_choice == "LSTM":
        st.info("""
        **LSTM + Self-Attention**
        
        - Учитывает последовательность слов
        - Механизм внимания
        - Хороший баланс скорости и точности
        """)
    else:
        st.info("""
        **RuBERT-tiny2**
        
        - Предобученная трансформер-модель
        - Понимает контекст и семантику
        - Наивысшая точность
        """)

# === БОКОВАЯ ПАНЕЛЬ ===
with st.sidebar:
    st.markdown("### ⚙️ Настройки")
    st.info("Модель загружена в память и кэширована.")
    
    if st.button("🔄 Сбросить состояние"):
        st.session_state.news_model_loaded = None
        st.session_state.news_model_data = {}
        st.session_state.news_prediction_result = None
        st.session_state.news_input_text = ""
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Последнее обновление: {datetime.now().strftime('%Y-%m-%d %H:%M')}")