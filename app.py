import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, logging
import seaborn as sns
import torch

# Configuración para reducir logs
logging.set_verbosity_error()

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones", layout="wide")
st.title("Análisis de Opiniones de Clientes")

# Descargar recursos de NLTK optimizado
nltk.download('punkt')
nltk.download('stopwords')

# Cargar modelos optimizados
@st.cache_resource
def load_models():
    try:
        # Modelo pequeño para análisis de sentimientos
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Modelo pequeño para resumen
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1
        )
        
        return sentiment_analyzer, summarizer
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None

sentiment_analyzer, summarizer = load_models()

# Funciones optimizadas
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def analyze_sentiment(text):
    if sentiment_analyzer is None:
        return "Error", 0.0
    try:
        result = sentiment_analyzer(text[:512])
        label = result[0]['label']
        return {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}.get(label, "Neutral"), result[0]['score']
    except:
        return "Error", 0.0

def generate_summary(text):
    if summarizer is None:
        return "Modelo no disponible"
    try:
        return summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except:
        return "Error generando resumen"

# Interfaz de usuario optimizada
def main():
    st.header("Carga tus opiniones")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file).head(20)  # Limitar a 20 opiniones
        
        with st.spinner("Procesando..."):
            # Análisis básico
            st.subheader("Análisis Rápido")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Nube de palabras**")
                tokens = clean_and_tokenize(' '.join(df.iloc[:, 0].astype(str)))
                wordcloud = WordCloud(width=400, height=200).generate(' '.join(tokens))
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(plt)
            
            with col2:
                st.write("**Sentimientos**")
                df['Sentimiento'] = df.iloc[:, 0].apply(lambda x: analyze_sentiment(x)[0])
                st.bar_chart(df['Sentimiento'].value_counts())
        
        # Análisis detallado bajo demanda
        if st.checkbox("Mostrar análisis detallado"):
            with st.spinner("Analizando..."):
                df['Detalle'] = df.iloc[:, 0].apply(lambda x: analyze_sentiment(x)[0] + f" ({analyze_sentiment(x)[1]:.2f})")
                st.dataframe(df[[df.columns[0], 'Detalle']])

if __name__ == "__main__":
    main()
