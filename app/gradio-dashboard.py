import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# ========== Load API Key from Hugging Face Secrets ==========
# Load file .env
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ========== Load & Prepare Book Dataset ==========
books = pd.read_csv('D://Workspace//LLM//Book Recommendation//dataset//books_with_emotions.csv')

# Add fallback thumbnail image for books with missing cover
books['large_thumbnail'] = books['thumbnail'] + '&file=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail']
)

# ==========  Load Vector Database for Semantic Search ==========

#Create embeddings and vector database
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

database_books = Chroma(
    persist_directory="D://Workspace//LLM//Book Recommendation//chroma",
    embedding_function=huggingface_embeddings
)

# ========== Semantic Retrieval Logic ==========
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Search top-N semantically similar books
    recs = database_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Apply category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

    else:
        book_recs = book_recs.head(final_top_k)

    # Apply emotion tone sorting
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs

# ========== Frontend Book Formatter ==========
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Truncate description
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Format authors nicely
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create caption and append result
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# ========== Gradio UI ==========
# Dropdown choices
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Define Gradio Blocks Interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.HTML("<h1 style='font-size: 48px; color: #5A189A; text-align: center;'>Book Recommendation System</h1>")
    gr.Markdown("Enter your query and get LLM-powered book suggestions based on meaning, tone, and genre.")

    with gr.Row():
        user_query = gr.Textbox(label="Describe your book interest:", placeholder="e.g., A story about forgiveness and healing")
        category_dropdown = gr.Dropdown(choices=categories, label="Genre Filter:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone Filter:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=3)

    gr.HTML("""
        <div style='text-align: center; margin-top: 60px;'>
            <p style='font-size: 18px; color: #444;'><strong>Nguyen Minh Tri</strong></p>
            <a href='https://www.linkedin.com/in/minh-tr%C3%AD-nguy%E1%BB%85n-16b845327/' target='_blank' style='color: #5A189A; font-weight: bold;'>Connect on LinkedIn</a>
        </div>
    """)

    # Connect button to function
    submit_button.click(fn= recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)


if __name__ == "__main__":
    dashboard.launch(share=True)