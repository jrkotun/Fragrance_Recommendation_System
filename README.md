# Fragrance Recommendation System
- Author: Habeeb Kotun Jr.
- Affiliation: School of Applied Computational Sciences, Meharry Medical College
- Email: hkotun24@mmc.edu

## Project Overview
The Fragrance Recommendation System is a Streamlit-based web application that delivers real-time, personalized fragrance suggestions based on user input. It employs a Retrieval-Augmented Generation (RAG) approach by combining FAISS indexing and LLM-generated explanations via Ollama. The system uses cosine similarity to find the most relevant fragrance entries based on user queries, retrieves them efficiently with FAISS, and then enahcnces the results with human-readable rationales. This approach helps users navigate the complex world of perfumes with tailored recommendations and interpretable insights.

## Live Demo
Check out the deployed app on Hugging Face Spaces:

## Key Features
- Natural language search: Enter your preferences like "fresh scent for springtime" or "dark, woody scent for evening wear" or a more complex query.
- Semantic embedding: Converts fragrance metadata into text and embeds them using a Sentence Transformer model.
- FAISS-powered retrieval: Fast nearest-neighbor search across thousands of fragrances.
- Structured filters: Refine results by brand, gender, rating, accords, or specific notes.
- LLM-generated rationale: Uses Ollama to explain why a particular fragrance was recommended.
- Fragrance cards: Displays perfume details with brand, perfumers, notes, accords, and the Ollama explanation.

## Project Structure
- Fragrance_Recommender.ipynb: Main notebook, contains the code to read in the fragrance data, create the FAISS index, and create the streamlit app.
- fra_cleaned.csv: Cleaned dataset with fragrance metadata
- fragrance_faiss.index: FAISS vector index
- fragrance_metadata.pkl: Fragrance metadata
- fragrance_recommendation_app.py: Main Streamlit application
- README.md: Project Documentation

## Dataset
Utilized a publicly available dataset from Kaggle that was originally scraped from Fragrantica.com and contains information on over 24,000 fragrances.
Dataset includes the following information for each fragrance:
- Fragrance name, house, country, release year
- Gender classification
- Ratings and review count
- Note pyramid (top, middle, base)
- Accords (e.g., citrus, amber, woody)
- Perfumer(s)

## Tech Stack
| Component       | Tool/Library              |
| --------------- | ------------------------- |
| Web UI          | Streamlit                 |
| Embeddings      | SentenceTransformers      |
| Semantic Search | FAISS                     |
| NLP Explanation | Ollama (local LLM engine) |
| Data Source     | Fragrantica (via Kaggle)  |
| Environment     | Python                    |

## Example Query and Response

### User Input
I’m looking for a unisex fragrance that’s suitable for both work and casual outings. It should be fresh, not too overpowering, and have a clean scent. Please show me options that fit these criteria.

### Top Fragrance Recommendation
- Name: Fragrance 04.
- House: Dedcool.
- Country: USA.
- Gender: unisex.
- Rating: 3.56 from 90 ratings.
- Year: 2018.
- Top Notes: black pepper, fig.
- Middle Notes: bergamot, lemon, jasmine, freesia.
- Base Notes: patchouli, incense, sandalwood.
- Perfumer: Carina Chaz.
- Accords: citrus, fresh spicy, woody, warm spicy, patchouli.

### Cosine Similarity Score: 0.572

### AI Explanation
Based on the user's query, Fragrance 04 by Dedcool was recommended because it meets the criteria of being a unisex fragrance suitable for both work and casual outings. The high cosine similarity score (0.572) indicates that the fragrance is quite close in terms of scent characteristics to what the user is looking for - fresh, clean, and not overpowering. This suggests that Fragrance 04's notes of citrus, bergamot, lemon, and patchouli align well with the user's preferences.
