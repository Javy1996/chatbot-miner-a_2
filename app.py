import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaEmbedding

# Obtener la clave de API desde Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Configuración del modelo LLM y embeddings
llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
embed_model = LlamaEmbedding(model="text-embedding-3-small", api_key=openai_api_key)

# Crear contexto de servicio
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

# Leer documentos desde la carpeta 'docs'
docs = SimpleDirectoryReader("docs").load_data()

# Crear el índice vectorial
index = VectorStoreIndex.from_documents(docs, service_context=service_context)

# Construir motor de consulta
query_engine = index.as_query_engine()

# Interfaz de Streamlit
st.title("Chatbot Minero")
query = st.text_input("Haz una pregunta sobre los documentos")

if query:
    response = query_engine.query(query)
    st.markdown(f"**Respuesta:** {response.response}")
