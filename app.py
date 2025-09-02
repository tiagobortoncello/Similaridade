import streamlit as st
import pandas as pd
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# O link do Google Drive para o seu arquivo CSV
CSV_URL = 'https://drive.google.com/uc?export=download&id=1fYroWa2-jgWIp6vbeTXYfpN76ev8fxSv'

# Título da aplicação
st.title('Gerador de Resumo e Indexação (Sem API)')
st.write('Cole um texto para gerar um resumo e encontrar os termos de indexação mais adequados, tudo com modelos de código aberto.')

# --- Carregamento e Preparação dos Dados ---
@st.cache_data
def load_data(file_url):
    try:
        df = pd.read_csv(file_url)
        # Use o 'texto' como a fonte de conhecimento para o resumo
        df['page_content'] = df['texto'].fillna('')
        return df
    except Exception as e:
        st.error(f'Erro ao carregar o arquivo de dados da URL: {e}')
        return None

df = load_data(CSV_URL)

# --- Carregamento dos Modelos ---
@st.cache_resource
def load_models():
    # Modelo para criar embeddings e buscar similaridade
    # 'distilbert-base-nli-stsb-mean-tokens' é um modelo pequeno e eficiente
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Modelo para resumo generativo
    # 't5-small' é um modelo pequeno que tem mais chance de rodar no Streamlit Cloud gratuito
    summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")

    return embedding_model, summarizer_tokenizer, summarizer_model

if df is not None:
    embedding_model, summarizer_tokenizer, summarizer_model = load_models()
    
    # 1. Pré-processa todos os textos do CSV para a busca por similaridade
    # st.cache_data armazena os embeddings para que não sejam gerados novamente
    @st.cache_data
    def get_embeddings(texts):
        return embedding_model.encode(texts, convert_to_tensor=True)

    corpus_embeddings = get_embeddings(df['page_content'].tolist())
    
    # --- Interface do Usuário ---
    user_text = st.text_area('Cole o seu texto aqui:', height=200)

    if st.button('Gerar Sugestões'):
        if user_text:
            # 2. Busca de Contexto (Recuperação)
            # Encontra o texto mais similar no CSV para usar como contexto
            query_embedding = embedding_model.encode(user_text, convert_to_tensor=True)
            similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), corpus_embeddings, dim=1)
            most_similar_index = similarities.argmax().item()
            most_similar_row = df.iloc[most_similar_index]
            
            # Pega o contexto do CSV
            context_text = most_similar_row['page_content']

            # 3. Geração do Resumo (Generativo com Hugging Face)
            # Constrói o prompt para o modelo T5
            prompt = f"summarize: {context_text}"
            inputs = summarizer_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
            
            # Gera o resumo
            outputs = summarizer_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            generated_summary = summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.success('Sugestões geradas com sucesso!')
            
            # Exibe o resumo gerado
            st.subheader('Resumo Gerado')
            st.write(generated_summary)
            
            # Exibe os termos de indexação encontrados (usando o mesmo índice)
            st.subheader('Termos de Indexação Sugeridos')
            if pd.isna(most_similar_row['termos de indexação']):
                st.write('Não há termos de indexação disponíveis para a entrada mais similar.')
            else:
                st.write(most_similar_row['termos de indexação'])

            st.markdown("---")
            st.info(f"Pontuação de similaridade para os termos: **{similarities[most_similar_index]:.2f}**")
        else:
            st.warning('Por favor, cole um texto na área acima para continuar.')
