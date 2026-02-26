import streamlit as st
from utils import page_header

st.set_page_config(
    page_title="NLP Evolution",
    page_icon="🧠",
    layout="wide"
)

page_header("Эволюция подходов в NLP", "🧠")

st.markdown("""
### 🎯 О проекте
Демонстрация развития методов обработки естественного языка: от классического ML до современных LLM с эффективной дообучкой.

### 📚 Структура демонстрации
| Страница | Технология | Задача | Статус |
|----------|------------|--------|--------|
| **Классификация отзывов** | TF-IDF + LSTM + BERT | Sentiment Analysis | 🔨 В разработке |
| **Классификация новостей** | Transformers | Topic Classification | 🔨 В разработке |
| **Генерация текста (LLM)** | Qwen + QLoRA (4-bit) | Text Generation | ✅ Готово |

### 🛠 Технологический стек
- **Frontend:** Streamlit
- **ML Framework:** PyTorch, HuggingFace Transformers
- **Fine-tuning:** PEFT, LoRA, QLoRA
- **Quantization:** bitsandbytes (4-bit)
""")

st.markdown("---")
st.caption("Разработано в рамках образовательного проекта по NLP")