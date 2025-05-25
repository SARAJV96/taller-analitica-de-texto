import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline
import seaborn as sns

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Configuración inicial
st.set_page_config(page_title="Análisis de Opiniones de Bases de Maquillaje", layout="wide")

# Título de la aplicación
st.title("Análisis de Opiniones - Bases de Maquillaje")

# Cargar modelos de Hugging Face
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment_analyzer, summarizer

sentiment_analyzer, summarizer = load_models()

# Función para limpiar y tokenizar texto
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Función para analizar sentimientos
def analyze_sentiment(text):
    result = sentiment_analyzer(text[:512])
    label = result[0]['label']
    if label == 'POS':
        return "Positivo", result[0]['score']
    elif label == 'NEG':
        return "Negativo", result[0]['score']
    else:
        return "Neutral", result[0]['score']

# Función para generar resumen
def generate_summary(text):
    summary = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Datos de ejemplo
opinions = [
    "Un sérum magnífico, deja la piel espectacular con un acabado natural, el tono está muy bien.",
    "Este producto es maravilloso, minimiza imperfecciones con una sola aplicación al día. 10/10.",
    "Es la mejor base si buscas una cobertura muy natural. No se nota que traes algo puesto.",
    "Excelente base buen cubrimiento.",
    "Mi piel es sensible y este producto es el mejor aliado del día a día, excelente cubrimiento.",
    "Excelente base buen cubrimiento.",
    "El empaque es terrible, no la volveré a comprar porque no sirve el envase.",
    "Sí se siente una piel diferente después de usar el producto.",
    "Me gusta mucho cómo deja mi piel, es buen producto aunque no me gusta su presentación.",
    "Me parece buena, pero pienso que huele mucho a alcohol, no sé si es normal.",
    "Creo que fue el color que no lo supe elegir, no está mal, pero me imaginaba algo más.",
    "La base ofrece un acabado mate y aterciopelado que deja la piel lisa.",
    "La base de maquillaje ofrece un acabado muy lindo y natural.",
    "Muy buen producto, solo que dura poco tiempo, por ahí unas 5 horas.",
    "Excelente cobertura y precio.",
    "No es para nada grasosa.",
    "El producto es mucho más oscuro de lo que aparece en la referencia.",
    "Pensé me sentaría mejor el número 8, es muy buena pero noto que toca como poner dos veces.",
    "No me gustó su cobertura.",
    "La sensación en la piel no me gusta, me arde al aplicarla."
]

df = pd.DataFrame({'Opinion': opinions})

# Procesamiento inicial
all_text = ' '.join(df['Opinion'].astype(str))
tokens = clean_and_tokenize(all_text)
df['Análisis'] = df['Opinion'].apply(analyze_sentiment)
df[['Sentimiento', 'Puntaje']] = pd.DataFrame(df['Análisis'].tolist(), index=df.index)

# Interfaz de usuario
st.header("Visualización de Datos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Nube de Palabras")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

with col2:
    st.subheader("Palabras Más Frecuentes")
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(10)
    top_words_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Frecuencia', y='Palabra', data=top_words_df, palette='viridis')
    st.pyplot(plt)

st.header("Análisis de Sentimientos")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribución de Sentimientos")
    sentiment_counts = df['Sentimiento'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#99ff99','#ffcc99'])
    plt.title('Distribución de Sentimientos')
    st.pyplot(plt)

with col4:
    st.subheader("Opiniones por Sentimiento")
    st.dataframe(df[['Opinion', 'Sentimiento', 'Puntaje']].sort_values('Puntaje', ascending=False))

st.header("Interacción con los Comentarios")

selected_opinion = st.selectbox("Selecciona una opinión para analizar:", df['Opinion'])

if st.button("Generar Resumen"):
    summary = generate_summary(selected_opinion)
    st.write("**Resumen:**", summary)

if st.button("Identificar Temas Principales"):
    topics = {
        "Cobertura": ["cobertura", "cubrimiento", "tapar", "imperfecciones"],
        "Textura": ["textura", "acabado", "aterciopelado", "mate", "natural"],
        "Duración": ["dura", "horas", "tiempo", "permanece"],
        "Color": ["color", "tono", "oscuro", "número"],
        "Piel sensible": ["sensible", "arda", "irrita", "reacción"]
    }
    
    matched_topics = []
    tokens = clean_and_tokenize(selected_opinion.lower())
    
    for topic, keywords in topics.items():
        if any(keyword in tokens for keyword in keywords):
            matched_topics.append(topic)
    
    if matched_topics:
        st.write("**Temas identificados:**", ", ".join(matched_topics))
    else:
        st.write("No se identificaron temas específicos en esta opinión.")