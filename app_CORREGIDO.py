import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# Obtener la clave API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Configuración de la página de Streamlit
st.set_page_config(page_title="Chatbot Minería", layout="centered")

st.title("🤖 Chatbot de Minería")
st.write("Haz una pregunta sobre los temas de la asignatura de minería:")

# Entrada del usuario
pregunta = st.text_input("Escribe tu pregunta")

if pregunta:
    # Cargar documentos desde la carpeta
    docs = SimpleDirectoryReader("docs_mineria").load_data()

    # Crear contexto de servicio con el modelo de OpenAI
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-4", api_key=api_key)
    )

    # Crear el índice vectorial
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)

    # Motor de consulta
    query_engine = index.as_query_engine()

    with st.spinner("Pensando..."):
        respuesta = query_engine.query(pregunta)
        st.success("Respuesta:")
        st.write(str(respuesta))  # forzar conversión a texto
else:
    st.info("Por favor, ingresa una pregunta.")
