import streamlit as st
import time

def page_header(title, icon):
    """Стандартный заголовок для страниц"""
    st.markdown(f"# {icon} {title}")
    st.markdown("---")

def create_placeholder_card(task_name, status="🔨 В разработке"):
    """Карточка-заглушка для незавершенных задач"""
    st.info(f"**Статус:** {status}")
    st.warning(f"⚠️ {task_name} находится в стадии реализации.")
    st.markdown("""
    **План реализации:**
    - [ ] Загрузка предобученных моделей
    - [ ] Инференс на тестовых данных
    - [ ] Визуализация метрик
    - [ ] Деплой интерфейса
    """)

def simulate_inference(delay=0.5):
    """Имитация задержки инференса для демо"""
    with st.spinner("⏳ Модель думает..."):
        time.sleep(delay)
    return True