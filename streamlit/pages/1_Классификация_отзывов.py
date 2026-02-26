import streamlit as st
from utils import page_header, create_placeholder_card, simulate_inference

st.set_page_config(page_title="Отзывы", page_icon="🏥")
page_header("Классификация отзывов на поликлиники", "🏥")

# Боковая панель для загрузки
with st.sidebar:
    st.header("⚙️ Настройки")
    model_choice = st.selectbox(
        "Выберите модель",
        ["Baseline (LogReg)", "LSTM", "BERT (rubert-tiny2)"],
        help="Модели-заглушки для демонстрации"
    )
    uploaded_file = st.file_uploader("Загрузить отзывы (CSV)", type=["csv"])

create_placeholder_card("Модели классификации тональности")

# Демонстрация интерфейса (без бэкенда)
st.subheader("📝 Тестирование")
user_text = st.text_area("Введите текст отзыва", "Врач был очень внимательным, спасибо!")

if st.button("Получить предсказание", type="primary"):
    if simulate_inference():
        st.success("✅ Тон: Позитивный (Демо)")
        st.info("⏱ Время инференса: ~50 мс (заглушка)")

# Место под графики
st.subheader("📊 Метрики моделей")
st.markdown("*(Здесь будет сводная таблица F1-macro и сравнение скоростей)*")
st.empty()