import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI as LlamaOpenAI
from llama_index.embeddings import OpenAIEmbedding as LlamaEmbedding

# Leer la API Key de OpenAI desde los secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Crear LLM y modelo de embedding con configuración mínima compatible
llm = LlamaOpenAI(
    model="gpt-3.5-turbo",
    api_key=openai_api_key
)

embed_model = LlamaEmbedding(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# Crear ServiceContext con ambos modelos
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

# Cargar documentos desde /docs
documents = SimpleDirectoryReader("docs").load_data()

# Crear índice vectorial
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Crear motor de consulta
query_engine = index.as_query_engine()

# Interfaz en Streamlit
st.title("🤖 Chatbot Minero")

question = st.text_input("¿Qué deseas saber?")

if question:
    with st.spinner("Pensando..."):
        response = query_engine.query(question)
        st.write(response.response)
