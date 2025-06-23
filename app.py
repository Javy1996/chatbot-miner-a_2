import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Leer API key desde secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Crear modelo LLM y de embedding
llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small")

# Contexto de servicio
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

# Leer documentos desde la carpeta 'docs'
docs = SimpleDirectoryReader("docs").load_data()

# Crear índice vectorial
index = VectorStoreIndex.from_documents(docs, service_context=service_context)

# Crear motor de preguntas
query_engine = index.as_query_engine()

# Interfaz Streamlit
st.title("🤖 Chatbot Minero")

user_question = st.text_input("Hazme una pregunta sobre los documentos:")

if user_question:
    with st.spinner("Pensando..."):
        response = query_engine.query(user_question)
        st.write(response.response)
