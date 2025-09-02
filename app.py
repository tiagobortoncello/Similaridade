import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch

# Verifique se o PyTorch está instalado corretamente
if not torch.cuda.is_available():
    st.info("Atenção: Não há GPU disponível. O modelo de resumo pode ser lento.")

# Link do Google Drive para o seu arquivo CSV
CSV_URL = 'https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv'

st.title('Gerador de Resumo Generativo e Indexação')
st.write('Cole um texto para gerar um resumo e encontrar os termos de indexação mais adequados.')

# --- Carregamento dos Dados ---
@st.cache_data
def load_data(file_url):
    try:
        df = pd.read_csv(file_url)
        df['texto_completo'] = df['resumo'].fillna('') + ' ' + df['texto'].fillna('')
        return df
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo de dados da URL: {e}')
        return None

df = load_data(CSV_URL)

# --- Carregamento do Modelo Generativo ---
# Use @st.cache_resource para modelos de ML para evitar recarregar
@st.cache_resource
def load_summarizer_model():
    # Modelos maiores como 't5-large' podem não funcionar no Streamlit Cloud gratuito
    try:
        summarizer = pipeline("summarization", model="t5-small", framework="pt")
        return summarizer
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de resumo: {e}")
        return None

if df is not None:
    summarizer = load_summarizer_model()
    
    # Cria o vetorizador para a busca dos termos de indexação
    vectorizer = TfidfVectorizer(stop_words=None) 
    tfidf_matrix = vectorizer.fit_transform(df['texto_completo'])

    # --- Interface do Usuário ---
    user_text = st.text_area('Cole o seu texto aqui:', height=200)

    if st.button('Gerar Sugestões'):
        if user_text and summarizer:
            # 1. Geração do Resumo (Tarefa Generativa)
            # Defina o max_length para controlar o tamanho do resumo
            generated_summary = summarizer(user_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            
            # 2. Busca de Termos de Indexação (Tarefa de Similaridade)
            user_text_vector = vectorizer.transform([user_text])
            cosine_similarities = cosine_similarity(user_text_vector, tfidf_matrix).flatten()
            most_similar_index = cosine_similarities.argmax()
            most_similar_row = df.iloc[most_similar_index]

            st.success('Sugestões geradas com sucesso!')

            # Exibe o resumo gerado
            st.subheader('Resumo Gerado')
            st.write(generated_summary)
                
            # Exibe os termos de indexação encontrados
            st.subheader('Termos de Indexação Sugeridos')
            if pd.isna(most_similar_row['termos de indexação']):
                st.write('Não há termos de indexação disponíveis para a entrada mais similar.')
            else:
                st.write(most_similar_row['termos de indexação'])

            st.markdown("---")
            st.info(f"Pontuação de similaridade para os termos: **{cosine_similarities[most_similar_index]:.2f}**")
        else:
            st.warning('Por favor, cole um texto na área acima para continuar.')
