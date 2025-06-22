
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import os

# Configurar API Key
api_key = st.secrets["OPENAI_API_KEY"]

# Configurar el modelo LLM de forma global
Settings.llm = OpenAI(model="gpt-4", api_key=api_key)

# Título en la app
st.set_page_config(page_title="Chatbot Minero", layout="wide")
st.title("Chatbot Minero - Explora tus documentos mineros 📄⛏️")

# Cargar documentos
docs = SimpleDirectoryReader("docs_mineria").load_data()

# Crear índice a partir de documentos
index = VectorStoreIndex.from_documents(docs)

# Crear motor de consulta
query_engine = index.as_query_engine()

# Interfaz de usuario
prompt = st.text_input("Haz tu pregunta sobre los documentos:")

if prompt:
    response = query_engine.query(prompt)
    st.markdown("### Respuesta")
    st.write(response.response)
