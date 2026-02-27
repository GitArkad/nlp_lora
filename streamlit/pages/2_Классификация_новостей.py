import streamlit as st
import torch
import time
import json
from utils import page_header

# === 🔥 ИНИЦИАЛИЗАЦИЯ SESSION_STATE (САМОЕ НАЧАЛО!) ===
# Это должно быть ВЫПОЛНЕНО до любого кода, который читает session_state
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_classified": 0,
        "by_category": {'мода': 0, 'технологии': 0, 'финансы': 0, 'крипта': 0, 'спорт': 0}
    }
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

# === Конфигурация ===
CATEGORIES = ['мода', 'технологии', 'финансы', 'крипта', 'спорт']
LABEL_MAPPING = {'мода': 0, 'технологии': 1, 'финансы': 2, 'крипта': 3, 'спорт': 4}
ID_TO_CATEGORY = {v: k for k, v in LABEL_MAPPING.items()}

st.set_page_config(page_title="Новости", page_icon="📰", layout="wide")
page_header("Классификация тематики новостей Telegram", "📰")

# === Сайдбар ===
with st.sidebar:
    st.header("⚙️ Настройки модели")
    
    model_source = st.radio(
        "Источник модели",
        ["Локальный файл", "Hugging Face Hub"],
        help="Выберите откуда загружать модель"
    )
    
    if model_source == "Локальный файл":
        model_path = st.text_input(
            "Путь к модели",
            placeholder="./models/news-classifier",
            help="Путь к папке с файлами модели (.pt, .bin, .safetensors)"
        )
    else:
        model_path = st.text_input(
            "Hugging Face Repo ID",
            placeholder="username/model-name",
            help="Например: gitarkad/news-classifier"
        )
    
    load_model_btn = st.button("📥 Загрузить модель", type="primary", use_container_width=True)
    
    if load_model_btn and model_path:
        with st.spinner("Загрузка модели..."):
            try:
                # TODO: Заменить на реальную загрузку вашей модели
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.eval()
                
                if torch.cuda.is_available():
                    model = model.to('cuda')
                
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
                st.session_state.model_path = model_path
                
                st.success("✅ Модель загружена!")
                
            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {e}")
                st.info("💡 Проверьте путь к модели и наличие файлов")
    
    elif st.session_state.model_loaded:
        st.success(f"✅ Модель: `{st.session_state.model_path}`")
        if st.button("🔄 Перезагрузить модель"):
            st.session_state.model_loaded = False
            if "model" in st.session_state:
                del st.session_state.model
            if "tokenizer" in st.session_state:
                del st.session_state.tokenizer
            st.rerun()
    
    st.divider()
    
    # Настройки инференса
    st.subheader("⚡ Параметры")
    top_k = st.slider("Показать топ категорий", 1, len(CATEGORIES), 3)
    show_all = st.checkbox("Показывать все категории в деталях", value=True)
    
    st.divider()
    
    # Статистика — безопасное чтение (теперь stats точно инициализирован)
    st.subheader("📊 Статистика сессии")
    stats = st.session_state.get("stats", {})
    total = stats.get("total_classified", 0)
    by_category = stats.get("by_category", {c: 0 for c in CATEGORIES})
    
    st.metric("Всего классифицировано", total)
    
    # Мини-диаграмма по категориям
    if any(by_category.values()):
        import pandas as pd
        stats_df = pd.DataFrame({
            "Категория": CATEGORIES,
            "Количество": [by_category[c] for c in CATEGORIES]
        })
        st.bar_chart(stats_df.set_index("Категория"), use_container_width=True)

# === Основной интерфейс ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Ввод текста")
    news_text = st.text_area(
        "Текст поста из Telegram",
        value=st.session_state.news_text,
        height=200,
        placeholder="Вставьте текст новостного поста для классификации...",
        help="Модель определит тематику: мода, технологии, финансы, крипта или спорт",
        key="news_text_input"
    )
    
    # Примеры для быстрого тестирования
    with st.expander("📋 Примеры текстов"):
        examples = {
            "мода": "Новая коллекция весна-лето уже в магазинах! Трендовые цвета и фасоны.",
            "технологии": "Apple представила новый чип M4 с улучшенной нейросетью для обработки фото.",
            "финансы": "ЦБ поднял ключевую ставку. Что ждать от ипотеки и вкладов?",
            "крипта": "Bitcoin пробил отметку $70K. Аналитики прогнозируют дальнейший рост.",
            "спорт": "Зенит обыграл Спартак в дерби со счётом 3:1. Голы и лучшие моменты."
        }
        for cat, text in examples.items():
            if st.button(f"🎯 {cat.capitalize()}", key=f"ex_{cat}"):
                st.session_state.news_text = text
                st.rerun()

with col2:
    st.subheader("⚡ Быстрые действия")
    
    classify_btn = st.button(
        "🔍 Определить категорию",
        type="primary",
        use_container_width=True,
        disabled=not news_text
    )
    
    clear_btn = st.button("🧹 Очистить", use_container_width=True)
    
    if clear_btn:
        st.session_state.news_text = ""
        if "last_result" in st.session_state:
            del st.session_state.last_result
        st.rerun()
    
    st.divider()
    st.info("💡 **Категории:**\n" + ", ".join([f"`{c}`" for c in CATEGORIES]))

# === Классификация ===
if classify_btn and news_text:
    if not st.session_state.model_loaded:
        st.warning("⚠️ Сначала загрузите модель в настройках слева!")
        st.info("💡 Введите путь к модели и нажмите «Загрузить модель»")
    else:
        with st.spinner("🤖 Анализируем текст..."):
            start_time = time.time()
            
            try:
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                
                # Токенизация
                inputs = tokenizer(
                    news_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                # Предсказание
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    scores = probabilities[0].cpu().tolist()
                
                # Маппинг: индекс → категория → скор
                labeled_scores = [(CATEGORIES[i], scores[i]) for i in range(len(CATEGORIES))]
                labeled_scores.sort(key=lambda x: x[1], reverse=True)
                
                inference_time = time.time() - start_time
                
                # Формируем результат
                result = {
                    "text": news_text,
                    "labels": [label for label, score in labeled_scores],
                    "scores": [score for label, score in labeled_scores],
                    "predicted_class": labeled_scores[0][0],
                    "confidence": labeled_scores[0][1],
                    "time": inference_time
                }
                
                # Сохраняем результат
                st.session_state.last_result = result
                
                # Обновляем статистику
                st.session_state.stats["total_classified"] += 1
                top_label = result['predicted_class']
                st.session_state.stats["by_category"][top_label] += 1
                
            except Exception as e:
                st.error(f"❌ Ошибка классификации: {e}")
                st.info("💡 Попробуйте сократить текст или перезагрузить модель")

# === Отображение результатов ===
if st.session_state.last_result:
    result_data = st.session_state.last_result
    
    st.divider()
    st.subheader("📊 Результаты классификации")
    
    # Верхняя категория с цветовой индикацией уверенности
    top_label = result_data['predicted_class']
    top_score = result_data['confidence']
    
    # Цветовая шкала уверенности
    if top_score >= 0.9:
        color = "🟢"
        status = "Высокая"
    elif top_score >= 0.7:
        color = "🟡"
        status = "Средняя"
    else:
        color = "🔴"
        status = "Низкая"
    
    col_result1, col_result2, col_result3, col_result4 = st.columns(4)
    with col_result1:
        st.metric("Категория", f"{color} {top_label}")
    with col_result2:
        st.metric("Уверенность", f"{top_score:.1%}")
    with col_result3:
        st.metric("Уровень", status)
    with col_result4:
        st.metric("Время", f"{result_data['time']:.2f} сек")
    
    # Визуализация всех категорий
    st.markdown("### 📈 Распределение вероятностей")
    
    # Берем только топ-k для графика
    display_count = min(top_k, len(result_data['labels']))
    display_labels = result_data['labels'][:display_count]
    display_scores = result_data['scores'][:display_count]
    
    # Столбчатая диаграмма
    import pandas as pd
    df = pd.DataFrame({
        "Категория": display_labels,
        "Уверенность": [f"{s:.1%}" for s in display_scores],
        "Score": display_scores
    })
    
    st.bar_chart(
        df.set_index("Категория")[["Score"]],
        use_container_width=True
    )
    
    # Детальная таблица
    with st.expander("📋 Все категории (подробно)"):
        full_df = pd.DataFrame({
            "Категория": result_data['labels'],
            "Уверенность": [f"{s:.1%}" for s in result_data['scores']],
            "ID класса": [LABEL_MAPPING[c] for c in result_data['labels']]
        })
        st.dataframe(full_df, use_container_width=True, hide_index=True)
        
        # Raw output для отладки
        with st.expander("🔧 Raw output (для разработчиков)"):
            st.json({
                "predicted_label": top_label,
                "predicted_id": LABEL_MAPPING[top_label],
                "all_scores": dict(zip(result_data['labels'], result_data['scores']))
            })

# === Статистика по категориям (внизу страницы) ===
st.divider()
st.subheader("📈 Статистика по категориям за сессию")

stats_by_cat = st.session_state.stats.get("by_category", {})
if any(stats_by_cat.values()):
    import pandas as pd
    
    stats_df = pd.DataFrame({
        "Категория": CATEGORIES,
        "Количество": [stats_by_cat.get(c, 0) for c in CATEGORIES],
        "ID": [LABEL_MAPPING[c] for c in CATEGORIES]
    })
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.bar_chart(
            stats_df.set_index("Категория")[["Количество"]],
            use_container_width=True,
            color="#4CAF50"
        )
    
    with col_chart2:
        st.dataframe(
            stats_df[["Категория", "Количество", "ID"]],
            use_container_width=True,
            hide_index=True
        )
else:
    st.info(f"📊 Классифицируйте тексты, чтобы увидеть статистику по категориям: {', '.join(CATEGORIES)}")

# === Футер ===
st.divider()
st.caption(f"📰 Категории: {', '.join([f'{c}({LABEL_MAPPING[c]})' for c in CATEGORIES])} | Fine-tuned модель")
# подсказки по замене То есть найти в коде и исрпавить c учетом заметок


# ❌ УДАЛИТЬ (Заглушка)
# # TODO: Заменить на реальную загрузку вашей модели
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# model.eval()

# if torch.cuda.is_available():
#     model = model.to('cuda')

# st.session_state.model = model
# st.session_state.tokenizer = tokenizer




# ✅ ВСТАВИТЬ (Реальная загрузка с проверкой)

# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# try:
#     # Загрузка токенизатора
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     # Загрузка модели с явным указанием классов
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_path,
#         num_labels=5,  # ← Важно: 5 классов!
#         id2label={0: 'мода', 1: 'технологии', 2: 'финансы', 3: 'крипта', 4: 'спорт'},
#         label2id={'мода': 0, 'технологии': 1, 'финансы': 2, 'крипта': 3, 'спорт': 4}
#     )
#     model.eval()
    
#     # Перенос на GPU
#     if torch.cuda.is_available():
#         model = model.to('cuda')
    
#     # Сохранение в сессию
#     st.session_state.model = model
#     st.session_state.tokenizer = tokenizer
#     st.session_state.model_loaded = True
#     st.session_state.model_path = model_path
    
#     st.success("✅ Модель загружена!")
    
# except Exception as e:
#     st.error(f"❌ Ошибка загрузки: {e}")
#     st.info("💡 Проверьте путь и наличие config.json")

# # --- УДАЛИТЬ ЭТОТ БЛОК ---
# # Заглушка для демонстрации
# import random
# scores = [random.random() for _ in categories]
# scores = [s / sum(scores) for s in scores]  # Нормализация
# sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
# # --------------------------
# Токенизация входного текста



# ✅ ВСТАВИТЬ (Реальный PyTorch код)
# inputs = st.session_state.tokenizer(
#     news_text, 
#     return_tensors="pt", 
#     truncation=True, 
#     padding=True,
#     max_length=512  # Максимальная длина последовательности
# )

# # Перенос входных данных на GPU
# if torch.cuda.is_available():
#     inputs = {k: v.to('cuda') for k, v in inputs.items()}

# # Предсказание
# with torch.no_grad():  # Отключаем расчет градиентов
#     outputs = st.session_state.model(**inputs)
#     # Применяем softmax для получения вероятностей
#     probabilities = torch.softmax(outputs.logits, dim=-1)
#     scores = probabilities[0].cpu().tolist()  # Конвертируем в список

# # Сортировка категорий по убыванию вероятности
# # categories берется из вашего списка CATEGORIES
# sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)



# ✅ Убедитесь, что список совпадает с моделью
# Должно быть ровно 5 классов в правильном порядке
CATEGORIES = ['мода', 'технологии', 'финансы', 'крипта', 'спорт']
LABEL_MAPPING = {'мода': 0, 'технологии': 1, 'финансы': 2, 'крипта': 3, 'спорт': 4}
