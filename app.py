import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS

# O link do Google Drive para o arquivo CSV
CSV_URL = 'https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv'

# Título da aplicação
st.title('Gerador de Resumo e Indexação (via RAG com Gemini)')
st.write('Cole um texto para gerar um resumo baseado no nosso banco de dados.')

# --- Carregamento dos Dados ---
@st.cache_data(ttl=3600)
def load_data(file_url):
    try:
        df = pd.read_csv(file_url)
        if 'resumo' not in df.columns or 'texto' not in df.columns:
            st.error("O arquivo CSV deve conter as colunas 'resumo' e 'texto'.")
            return None
        df['page_content'] = df['resumo'].fillna('') + ' ' + df['texto'].fillna('')
        return df
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo de dados da URL: {e}')
        return None

df = load_data(CSV_URL)

# --- Configuração do LangChain ---
@st.cache_resource
def setup_langchain():
    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Chave da API do Gemini não encontrada.")
            return None, None
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GEMINI_API_KEY"])
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GEMINI_API_KEY"])
        loader = DataFrameLoader(df, page_content_column="page_content")
        docs = loader.load()
        docsearch = FAISS.from_documents(docs, embeddings)
        chain = load_qa_chain(llm, chain_type="stuff")
        return docsearch, chain
    except Exception as e:
        st.error(f"Erro ao configurar o LangChain: {e}")
        return None, None

if df is not None:
    docsearch, chain = setup_langchain()
    if docsearch is None or chain is None:
        st.error("Falha na inicialização do LangChain. Verifique a chave da API ou os dados carregados.")
    else:
        user_text = st.text_area('Cole o seu texto aqui:', height=200, help="O texto para o qual você quer um resumo.")
        if st.button('Gerar Sugestão'):
            if user_text:
                if len(user_text) > 5000:
                    st.warning("O texto inserido é muito longo. Por favor, use até 5000 caracteres.")
                else:
                    try:
                        docs = docsearch.similarity_search(user_text)
                        prompt = (
                            "Gere um resumo conciso (máximo de 100 palavras) do texto fornecido, "
                            "usando o contexto do banco de dados para garantir precisão e relevância: "
                        )
                        response = chain.run(input_documents=docs, question=prompt + user_text)
                        generated_summary = response
                        st.success('Sugestão gerada com sucesso!')
                        st.subheader('Resumo Gerado')
                        st.write(generated_summary)
                    except Exception as e:
                        st.error(f"Ocorreu um erro: {e}")
                        st.warning("Verifique se sua chave de API do Gemini está correta.")
            else:
                st.warning('Por favor, cole um texto na área acima para continuar.')
else:
    st.error("Não foi possível carregar os dados do CSV. Verifique a URL ou o formato do arquivo.")
