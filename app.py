
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
import os

st.set_page_config(page_title="Chatbot Minería", layout="centered")

st.title("🤖 Chatbot de Minería")
st.write("Haz una pregunta sobre los temas de la asignatura de minería:")

pregunta = st.text_input("Escribe tu pregunta")

if pregunta:
    # Preparar entorno de LLM
    docs = SimpleDirectoryReader("docs_mineria").load_data()
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query_engine = index.as_query_engine()

    with st.spinner("Pensando..."):
        respuesta = query_engine.query(pregunta)
        st.success("Respuesta:")
        st.write(respuesta)
else:
    st.info("Por favor, ingresa una pregunta.")
