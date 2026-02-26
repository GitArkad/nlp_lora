import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# === Конфигурация ===
MODEL_ID = "Qwen/Qwen2-7B-Instruct"
ADAPTER_PATH = "/workspace/lora_project/qwen-cringe-lora"  # Абсолютный путь!

st.set_page_config(page_title="🔍 LoRA Debug", page_icon="🧐", layout="wide")

# === 🔥 ИНИЦИАЛИЗАЦИЯ SESSION_STATE (ВСЕ переменные!) ===
# Это должно быть ВЫПОЛНЕНО до любого кода, который читает session_state
if "base_model" not in st.session_state:
    st.session_state.base_model = None
if "lora_model" not in st.session_state:
    st.session_state.lora_model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "lora_status" not in st.session_state:
    st.session_state.lora_status = "⏳ Не загружен"
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Загрузка модели (кэшируется) ===
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.eval()
    
    lora_model = None
    lora_status = "❌ Не найден"
    try:
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        lora_model.eval()
        lora_status = "✅ ЗАГРУЖЕН"
    except Exception as e:
        lora_status = f"❌ Ошибка: {str(e)[:50]}"
    
    return base_model, lora_model, tokenizer, lora_status

# === Загрузка моделей (только один раз) ===
if not st.session_state.models_loaded:
    with st.spinner("Загрузка моделей..."):
        base, lora, tok, status = load_model()
        st.session_state.base_model = base
        st.session_state.lora_model = lora
        st.session_state.tokenizer = tok
        st.session_state.lora_status = status
        st.session_state.models_loaded = True
        st.success("✅ Модели готовы!")

# === Вспомогательная функция генерации ===
def generate_response(model, prompt_text, system_prompt, max_tokens=100, temperature=0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    
    text = st.session_state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = st.session_state.tokenizer(text, return_tensors="pt").to(model.device)
    
    # Включаем адаптер если это PeftModel
    if hasattr(model, "enable_adapter"):
        model.enable_adapter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,          # ↑ Было 100, ставим 300-500 для длинных ответов
            min_new_tokens=50,           # ↓ Запрещаем слишком короткие ответы
        
            # === Креативность ===
            temperature=0.8,             # ↑ 0.7-0.9 для баланса между креативом и связностью
            top_p=0.9,                   # Ядро выборки: отбрасывает маловероятные токены
            top_k=50,                    # Ограничивает выбор топ-50 токенов на шаг
        
        # === Борьба с повторами ===
            repetition_penalty=1.15,     # ↑ Штраф за повторение фраз (1.1-1.2)
            no_repeat_ngram_size=3,      
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )
    
    return st.session_state.tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

# === СТАТУСНАЯ ПАНЕЛЬ ===
st.title("🔍 LoRA Debug & Чат")
col1, col2, col3 = st.columns(3)

with col1:
    status_base = "✅ OK" if st.session_state.base_model is not None else "❌ Нет"
    st.metric("Базовая модель", status_base)
with col2:
    st.metric("LoRA Адаптер", st.session_state.lora_status)
with col3:
    is_peft = isinstance(st.session_state.lora_model, PeftModel) if st.session_state.lora_model else False
    st.metric("Тип модели", "PeftModel" if is_peft else "Base Only")

# Предупреждение если LoRA не загружен
if not is_peft and st.session_state.lora_status.startswith("❌"):
    st.error(f"⚠️ {st.session_state.lora_status}")
    st.code(f"Проверьте путь: {ADAPTER_PATH}", language="bash")

st.divider()

# === A/B ТЕСТ: Сравнение Base vs LoRA ===
st.subheader("🧪 A/B Тест: База vs LoRA")
test_prompt = st.text_input("Тестовый запрос", "как дела?")

if st.button("🚀 Сравнить"):
    if test_prompt and st.session_state.tokenizer:
        col_base, col_lora = st.columns(2)
        
        # 1. Base Model
        with col_base:
            st.markdown("#### 🤖 Base (обычный стиль)")
            with st.spinner("Генерация..."):
                base_ans = generate_response(
                    st.session_state.base_model,
                    test_prompt,
                    "Ты полезный ассистент."
                )
            st.info(base_ans)
        
        # 2. LoRA Model
        with col_lora:
            st.markdown("#### 🤪 LoRA (кринж стиль)")
            if is_peft:
                with st.spinner("Генерация..."):
                    lora_ans = generate_response(
                        st.session_state.lora_model,
                        test_prompt,
                        "Ты — ультра-хайповый зумер. Отвечай на сленге: имба, кринж, рофл, зашквар, вайб."
                    )
                st.success(lora_ans)
            else:
                st.warning("LoRA не загружен")
    else:
        st.warning("Введите запрос и дождитесь загрузки моделей")

st.divider()

# === ОБЫЧНЫЙ ЧАТ ===
st.subheader("💬 Чат с моделью")

# Отображение истории
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Ввод пользователя
if prompt := st.chat_input("Напиши что-нибудь..."):
    if not st.session_state.tokenizer:
        st.error("⏳ Модели ещё загружаются, подождите...")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔"):
            # Выбираем модель: LoRA если есть и работает, иначе база
            model = st.session_state.lora_model if is_peft else st.session_state.base_model
            
            response = generate_response(
                model,
                prompt,
                "Ты — ультра-хайповый зумер. Отвечай на сленге: имба, кринж, рофл, зашквар, вайб."
            )
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# === Футер ===
st.caption("🤪 Кринж-модель | Qwen-7B + LoRA | Local inference")