import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')

path = 'movies_metadata.csv'
ndf = pd.read_csv(path)
df = ndf[["title", "overview"]]
df.dropna()

df['combined_text'] = df['title'] + " " + df['overview']

try:
    with open('encoded_movies.pkl', 'rb') as file: 
        df = pickle.load(file) 
    print("Embeddings loaded successfully!")
except FileNotFoundError:
    print("The file 'encoded_movies.pkl' does not exist. Please compute and save the embeddings first.")

summarizer = pipeline("summarization")

def search_movies_with_scores(query, df, top_k=5):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = df.iloc[top_indices][['title', 'overview']].copy()
    results['similarity_score'] = similarities[top_indices]
    return results

# Streamlit UI
st.title("Semantic Movie Search")
st.write("Search for movies based on their themes or descriptions using semantic embeddings")

# Input box for user query
query = st.text_input("Enter your search query:")

# Display results when a query is entered
if query:
    results = search_movies_with_scores(query, df)
    st.write("## Results:")
    
    for _, row in results.iterrows():
        # Summarize the movie overview using Hugging Face summarizer
        summary = summarizer(row['overview'], max_length=200, min_length=10, do_sample=False)
        summarized_text = summary[0]['summary_text'] if summary else row['overview']  # Fallback to original if no summary
        
        st.write(f"**Title:** {row['title']}  \n*Similarity Score:* {row['similarity_score']:.2f}  \n*Overview:* {summarized_text}")
        st.write("---")