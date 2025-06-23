import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import ServiceContext

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-3.5-turbo")

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

@st.cache_resource
def load_index():
    if os.path.exists("storage"):
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        return load_index_from_storage(storage_context)
    else:
        docs = SimpleDirectoryReader("docs").load_data()
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        index.storage_context.persist(persist_dir="storage")
        return index

st.set_page_config(page_title="Chatbot Minero", layout="centered")

st.title("🤖 Chatbot Minero Inteligente")
st.markdown("Haz tus preguntas sobre documentos mineros cargados al sistema.")

query = st.text_input("Escribe tu pregunta aquí")

if query:
    with st.spinner("Pensando..."):
        index = load_index()
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)
        st.write("📝 Respuesta:")
        st.markdown(response.response)
