
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from langchain_ollama import ChatOllama

# Fragrance card function
def create_fragrance_card(name, rating, brand, perfumer_text, top_notes, middle_notes, base_notes, accords_text, similarity_score, explanation):
    # Create fragrance card HTML
    card_html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 15px; 
                    background: linear-gradient(to bottom right, #ffffff, #f2f6fc); 
                    width: 400px; color: #222; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #3a3a3a; text-align: center;">{name} ⭐{rating}</h3>
            <p><strong>🏷️ Brand:</strong> {brand}</p>
            <p><strong>👃 Perfumer(s):</strong> {perfumer_text}</p>
            <p><strong>🌿 Top Notes:</strong> {top_notes}</p>
            <p><strong>💖 Heart Notes:</strong> {middle_notes}</p>
            <p><strong>🌲 Base Notes:</strong> {base_notes}</p>
            <p><strong>🎼 Main Accords:</strong> {accords_text}</p>
            <p><strong>🔎 Similarity Score:</strong> {similarity_score:.3f}</p>
            <p><strong>💡 AI Explanation:</strong> {explanation}</p>
        </div>
    """
    
    return card_html

# Load FAISS database, metadata, and encoder with cache
@st.cache_resource
def load_resources():
    index = faiss.read_index('fragrance_faiss.index')
    with open('fragrance_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    encoder = SentenceTransformer('paraphrase-mpnet-base-v2')
    return index, metadata, encoder

# Gets a brief explanation from Ollama for why this fragrance matches the user's query
def get_ollama_explanation(query, description, similarity):
    prompt = f"""
            A user is searching for a fragrance with this description: "{query}"
            
            One recommendation is:
            {description}
            
            The cosine similarity score between the user's query and this fragrance is {similarity:.3f}.
            
            Explain in 1-2 sentences, in plain English, why this fragrance was matches the user's query.
            """
    response = llm.invoke(prompt)
    return response.content.strip()

# Load Ollama
llm = ChatOllama(model="llama3.2")

# Initialize app
st.set_page_config(page_title="Fragrance Recommendation System", layout="wide")

# Add title to top of app interface
st.title("Fragrance Recommendation System")

# Sidebar filters
st.sidebar.header("Filters")
query = st.text_input("Describe your ideal fragrance:")

col1, col2 = st.columns(2)
with col1:
    k = st.slider("Number of recommendations:", 1, 10, 5)
with col2:
    min_rating = st.slider("Minimum rating:", 1.0, 5.0, 3.5)

gender_filter = st.sidebar.selectbox("Gender:", ["All", "Male", "Female", "Unisex"])
brand_filter = st.sidebar.text_input("Brand (leave empty for all):", "").title()
note_filter = st.sidebar.text_input("Notes (comma-separated):", "").lower()

# Load resources
index, metadata, encoder = load_resources()

# Convert rating_values to numeric
if 'rating_value' in metadata.columns:
    metadata['rating_value'] = pd.to_numeric(
        metadata['rating_value'], 
        errors='coerce')

# Press button and start recommendations
if st.button('Get Recommendations'):
    with st.spinner('Finding your fragrance recs...'):
        if query == "":
            st.warning("No query entered.")
        else:
            # Apply filters sequentially
            current_df = metadata.copy()
            
            # Gender filter
            if gender_filter != "All":
                current_df = current_df[current_df['gender'].str.lower() == gender_filter.lower()]
            
            # Brand filter
            if brand_filter:
                current_df = current_df[current_df['brand'].str.contains(brand_filter, case=False, na=False)]
            
            # Rating filter (with NaN handling)
            if 'rating_value' in current_df.columns:
                current_df = current_df[current_df['rating_value'].ge(min_rating)]
            
            # Note filter
            if note_filter:
                notes = [n.strip().lower() for n in note_filter.split(",")]
                def note_check(row):
                    note_fields = [
                        str(row['top']).lower() if pd.notna(row['top']) else "",
                        str(row['middle']).lower() if pd.notna(row['middle']) else "",
                        str(row['base']).lower() if pd.notna(row['base']) else ""
                    ]
                    return any(note in field for note in notes for field in note_fields)
                
                current_df = current_df[current_df.apply(note_check, axis=1)]
            
            valid_indices = current_df.index.tolist()
            
            # Check if any fragrances remain
            if not valid_indices:
                st.warning("No fragrances match all your filters. Try relaxing some criteria.")
                st.stop()

            # Grab the vectors for fragrances still present after the filters
            filtered_vectors = np.vstack([index.reconstruct(int(idx)) for idx in valid_indices])
            temp_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
            temp_index.add(filtered_vectors)
        
            # Encode the query and normalize it for cosine similarity
            query_vector = encoder.encode([query])
            faiss.normalize_L2(query_vector)
            
            # Perform the search and returns indices of the most similar vectors and their similarity scores
            sim_score, I = temp_index.search(query_vector, min(k, len(valid_indices)))

            # Get the recommened fragrance's indices and similarity score
            results = [(valid_indices[i], sim_score[0][j]) for j, i in enumerate(I[0])]
    
            # Display results
            st.subheader(f"Recommended Fragrances ({len(results)} results)")
            cols = st.columns(3)
            
            for idx, (result_idx, sim_score) in enumerate(results):
                rec = metadata.loc[result_idx]
    
                # Extract data with fallbacks
                name = rec.get('perfume', 'Unknown')
                brand = rec.get('brand', 'Unknown')
                perfumer_text = rec.get('perfumer', 'Unknown')
                top_notes = rec.get('top', 'Unknown')
                middle_notes = rec.get('middle', 'Unknown')
                base_notes = rec.get('base', 'Unknown')
                accords_text = rec.get('accord', 'Unknown')
                rating = rec.get('rating_value', '?')
    
                 # Create natural language fragrance description
                description = (
                    f"The fragrance is called {name}. It is by {brand}. "
                    f"The perfumer is {perfumer_text}. The top notes are {top_notes}, "
                    f"the heart notes are {middle_notes}, and the base notes are {base_notes}. "
                    f"The main accords are {accords_text}."
                )
    
                explanation = get_ollama_explanation(query, description, sim_score)
                
                # Add rating to card
                card = create_fragrance_card(
                        name,
                        rating,
                        brand, 
                        perfumer_text, 
                        top_notes,
                        middle_notes, 
                        base_notes, 
                        accords_text,
                        sim_score,
                        explanation
                    )
                cols[idx % 3].markdown(card, unsafe_allow_html=True)
