import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from PIL import Image
import pytesseract, fitz

# === 1. Tải dữ liệu và tạo chỉ mục index ===
@st.cache_resource
def load_index():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", temperature=0.3))
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

index = load_index()

# === 2. UI ===
st.set_page_config(page_title="Chatbot LSCM", layout="centered")
st.title("📦 Chatbot Giải Bài Tập LSCM")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === 3. Upload file ===
st.subheader("\U0001F4C2 Tải bài tập (PDF/Ảnh)")
uploaded_file = st.file_uploader("Chọn file PDF hoặc Ảnh", type=["pdf", "png", "jpg", "jpeg"])

extracted_text = ""

if uploaded_file:
    file_type = uploaded_file.type
    if "pdf" in file_type:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            extracted_text += page.get_text()
    elif "image" in file_type:
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image, lang="eng+vie")
    
    if extracted_text:
        st.success("Đã trích xuất được nội dung")
        st.text_area("Nội dung:", extracted_text, height=200)

# === 4. Hỏi đáp ===
st.subheader(":speech_balloon: Đặt câu hỏi")
user_question = st.text_input("Nhập câu hỏi về bài tập:")

if user_question:
    st.session_state.chat_history.append(("Bạn", user_question))
    response = index.as_chat_engine(chat_mode="condense_question").chat(user_question)
    st.session_state.chat_history.append(("Chatbot", response))

# === 5. Hiển thị lịch sử chat ===
if st.session_state.chat_history:
    st.subheader(":arrows_counterclockwise: Lịch sử chat")
    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
