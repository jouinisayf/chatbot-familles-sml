import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Titre
st.title("🔍 Prédiction de famille de produit à partir d'une description client")

# Chargement du fichier Excel
@st.cache_data
def load_data():
    df = pd.read_excel("Famille-description.xlsx")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Vérification des colonnes nécessaires
if not {'Famille', 'Des', 'Description'}.issubset(df.columns):
    st.error("❌ Le fichier Excel doit contenir les colonnes : 'Famille', 'Des' et 'Description'")
    st.stop()

# Chargement du modèle
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

modele = load_model()

# Encodage des références
@st.cache_data
def encode_descriptions(df):
    textes = (df['Des'].astype(str) + " " + df['Description'].astype(str)).tolist()
    return modele.encode(textes, convert_to_tensor=True)

embeddings_ref = encode_descriptions(df)

# Zone de saisie utilisateur
texte = st.text_area("📝 Entrez une description client :", height=150)

if texte:
    embedding_query = modele.encode(texte, convert_to_tensor=True)
    similarites = util.cos_sim(embedding_query, embeddings_ref)[0]
    indices_tries = similarites.argsort(descending=True)

    st.subheader("📊 Familles classées par similarité :")
    for idx in indices_tries:
        famille = df.iloc[int(idx)]['Famille']
        score = float(similarites[idx]) * 100
        st.write(f"- {famille} : **{score:.2f}%**")
