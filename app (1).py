import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaEmbedding
# Obtener clave desde secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
# Cargar documentos
documents_path = "docs"
docs = SimpleDirectoryReader(documents_path).load_data()
# Configurar modelos LLM y de embedding
llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
embed_model = LlamaEmbedding(model="text-embedding-ada-002", api_key=openai_api_key)
# Crear contexto
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
# Crear índice
index = VectorStoreIndex.from_documents(docs, service_context=service_context)
# Crear interfaz
st.title("Chatbot de Minería - Proyecto")
query = st.text_input("Haz una pregunta sobre los documentos:")
if query:
    response = index.as_query_engine().query(query)
    st.write("Respuesta:")
    st.write(response)
