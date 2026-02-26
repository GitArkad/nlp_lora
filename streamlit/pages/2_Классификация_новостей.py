import streamlit as st
from utils import page_header, create_placeholder_card, simulate_inference

st.set_page_config(page_title="Новости", page_icon="📰")
page_header("Классификация тематики новостей Telegram", "📰")

create_placeholder_card("Классификация тем новостей")

st.subheader("📝 Ввод текста")
news_text = st.text_area("Текст поста из Telegram", "В Москве открылся новый парк...")

if st.button("Определить категорию", type="primary"):
    if simulate_inference():
        st.success("✅ Категория: Общество (Демо)")
        st.json({"confidence": 0.85, "all_scores": {"Общество": 0.85, "Политика": 0.10, "Спорт": 0.05}})

st.subheader("📈 Статистика по категориям")
st.markdown("*(Здесь будет распределение предсказанных тем)*")
st.empty()