import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# O link do Google Drive para o seu arquivo CSV
# O formato 'uc?export=download' é essencial para o carregamento direto
CSV_URL = 'https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv'

# Título da aplicação
st.title('Gerador de Resumo e Indexação')
st.write('Cole um texto para encontrar o resumo e os termos de indexação mais adequados do nosso banco de dados.')

# --- Carregamento e Preparação dos Dados ---
# Tenta carregar o arquivo CSV. st.cache_data armazena o resultado para que o dataframe não seja recarregado a cada interação,
# o que é crucial para otimizar o desempenho.
@st.cache_data
def load_data(file_url):
    try:
        # Carrega o CSV diretamente da URL
        df = pd.read_csv(file_url)
        
        # Combina as colunas para criar um único texto para a busca de similaridade.
        # Ajuste os nomes das colunas se eles forem diferentes no seu CSV.
        df['texto_completo'] = df['resumo'].fillna('') + ' ' + df['texto'].fillna('')
        return df
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo de dados da URL: {e}')
        return None

df = load_data(CSV_URL)

if df is not None:
    # Cria o modelo TF-IDF e o vetorizador
    vectorizer = TfidfVectorizer(stop_words=None) 
    tfidf_matrix = vectorizer.fit_transform(df['texto_completo'])

    # --- Interface do Usuário ---
    # Área para o usuário colar o texto
    user_text = st.text_area('Cole o seu texto aqui:', height=200)

    if st.button('Gerar Sugestões'):
        if user_text:
            # Vetoriza o texto do usuário
            user_text_vector = vectorizer.transform([user_text])
            
            # Calcula a similaridade de cosseno
            cosine_similarities = cosine_similarity(user_text_vector, tfidf_matrix).flatten()
            
            # Encontra o índice do texto mais similar
            most_similar_index = cosine_similarities.argmax()
            
            # Pega a linha mais similar do dataframe
            most_similar_row = df.iloc[most_similar_index]
            
            # Exibe os resultados
            st.success('Sugestões geradas com sucesso!')

            # Exibe o resumo sugerido
            st.subheader('Resumo Sugerido')
            if pd.isna(most_similar_row['resumo']):
                st.write('Não há resumo disponível para a entrada mais similar.')
            else:
                st.write(most_similar_row['resumo'])
                
            # Exibe os termos de indexação sugeridos
            st.subheader('Termos de Indexação Sugeridos')
            if pd.isna(most_similar_row['termos de indexação']):
                st.write('Não há termos de indexação disponíveis para a entrada mais similar.')
            else:
                st.write(most_similar_row['termos de indexação'])

            # Opcional: Mostra a pontuação de similaridade
            st.markdown(f"---")
            st.info(f"Pontuação de similaridade: **{cosine_similarities[most_similar_index]:.2f}**")
        else:
            st.warning('Por favor, cole um texto na área acima para continuar.')
