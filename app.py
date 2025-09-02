import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# O link do Google Drive para o seu arquivo CSV
CSV_URL = 'https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv'

# Título da aplicação
st.title('Gerador de Resumo e Indexação (via RAG com Gemini)')
st.write('Cole um texto para gerar um resumo baseado no nosso banco de dados.')

# --- Carregamento dos Dados ---
@st.cache_data
def load_data(file_url):
    try:
        df = pd.read_csv(file_url)
        # Combina as colunas para o modelo usar como contexto
        df['page_content'] = df['resumo'].fillna('') + ' ' + df['texto'].fillna('')
        return df
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo de dados da URL: {e}')
        return None

df = load_data(CSV_URL)

# --- Configuração do LangChain ---
# A anotação @st.cache_resource é essencial para o desempenho.
@st.cache_resource
def setup_langchain():
    # Use um modelo de embeddings do Hugging Face para evitar o erro assíncrono do Gemini.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # A API do Gemini pode ser usada no LLM sem problemas.
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GEMINI_API_KEY"])

    loader = DataFrameLoader(df, page_content_column="page_content")
    docs = loader.load()

    docsearch = FAISS.from_documents(docs, embeddings)
    
    chain = load_qa_chain(llm, chain_type="stuff")

    return docsearch, chain

if df is not None:
    docsearch, chain = setup_langchain()
    
    user_text = st.text_area('Cole o seu texto aqui:', height=200, help="O texto para o qual você quer um resumo.")

    if st.button('Gerar Sugestão'):
        if user_text:
            try:
                # Busca de Contexto no CSV (Recuperação)
                docs = docsearch.similarity_search(user_text)
                
                # Geração do Resumo (Generativo)
                prompt = "Gere um resumo do seguinte texto, usando o contexto fornecido para ser mais preciso: "
                response = chain.invoke({"input_documents": docs, "question": prompt + user_text})
                
                generated_summary = response["output_text"]
                
                st.success('Sugestão gerada com sucesso!')
                st.subheader('Resumo Gerado')
                st.write(generated_summary)
                
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
                st.warning("Verifique se sua chave de API do Gemini está correta.")
        else:
            st.warning('Por favor, cole um texto na área acima para continuar.')
