# Book Recommendation System With LLM

![UI](https://github.com/Swuzz123/Book-Recommendation/blob/master/demo/UI_demo.png)

## Introduction
With the rapid increase in online content and widespread access to Internet-connected devices, users are increasingly overwhelmed by the sheer volume of information available. This overload makes it challenging for them to decide which content to engage with or which products to buy. This project aims to support an online book retailer that has seen a drop in sales, largely because customers are struggling with too many options and finding it hard to choose which books to purchase.

A personalized and explainable book recommendation system built with LLMs, Hugging Face embeddings, and Langchain.
Unlike traditional methods based on ratings or collaborative filtering, it leverages book metadata (descriptions), **LLM-based classification**, **emotion detection**, and **semantic matching using Hugging Face embeddings** to recommend books more intelligently.

## Project Objective
To develop a fully explainable **language-based book recommendation** engine that:
- Responds to natural language queries (e.g., “Books about space exploration for children”)
- Converts book descriptions into vector representations using Hugging Face embeddings
- Retrieves semantically relevant books via the Chroma vector database
- Classifies books into Fiction, Nonfiction, or Children’s categories using zero-shot LLM classification
- Detects emotional tone in book content using a fine-tuned emotion detection model
- Provides an interactive Gradio interface for exploring books by genre, emotion, and meaning

## Project Progress

### 1. Data Cleaning & Exploration [`data-exploration.ipynb`](./data-exploration.ipynb)

* Loaded the [7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset from Kaggle
* Filtered books with:

  * Complete descriptions (≥25 words)
  * Valid authors and titles
* Merged title and subtitle
* Engineered features:

  * `title_and_subtitle`, `age_of_book`, `tagged_description` (used for vector indexing)
* Output saved as `books_cleaned.csv`

### 2. Semantic Embedding & Vector Search [`vector-search.ipynb`](./vector-search.ipynb)

* Embedded book descriptions using Huggingface's [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* Split text into chunks using LangChain's `CharacterTextSplitter`
* Stored embeddings in **ChromaDB**
* Enabled vector-based semantic retrieval:

```
retrieve_semantic_recommendation("a book has some love stories")
```

* Mapped retrieved vectors to book metadata using `isbn13`

### 3. Genre Classification via Zero-Shot LLM [`text-classification.ipynb`](./text-classification.ipynb)

* Original dataset had **479+ inconsistent categories**
* Reduced to 3 major genres:

  * `Fiction`
  * `Nonfiction`
  * `Children's Nonfiction`
* Used [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) with zero-shot classification
* Performance on validation set:

  *  Accuracy: **77.8%**
  * F1 Scores: 0.75 (Fiction), 0.80 (Nonfiction)
* Classified remaining uncategorized books
* Output saved as `books_with_categories.csv`

### 4. Emotion Detection for Tone Filtering [`sentiment-analysis.ipynb`](./sentiment-analysis.ipynb)

* Used [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — a fine-tuned transformer model
* Target emotions:

  * `joy`, `sadness`, `fear`, `anger`, `disgust`, `surprise`, `neutral`
* Strategy:

  * Split each description into sentences
  * Run classifier on each sentence
  * Aggregate scores using **max pooling**
* Merged scores with original dataset
* Final dataset saved as `books_with_emotions.csv`

## Prerequisites
- Python 3.10+ installed on your system
- A Hugging Face account with an access token
- A `.env` fie configured with your Hugging Face access token

## Configuration
1. Create a Hugging Face account
   - If you don’t have one, sign up at: [https://huggingface.co/join](https://huggingface.co/join)

2. Generate an Access Token
   - Go to your [Hugging Face settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Click on “New token”, choose the role as “Read”, and then copy the token.
     
3. Create a .env file in the root directory of the project
   - This file will store your token securely.

  Your `.env` file will look like this:
  ```
  # Hugging Face access token
  HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  ```
## Installation and Run app
#### 1 Clone the Repository:
```
git clone https://github.com/Swuzz123/Book-Recommendation.git
cd Book-Recommendation
```
#### 2. Set Hugging Face Access Token
Create a file `.env` in the root directory and add your Hugging Face access token:
```
echo "HF_TOKEN=your-huggingface-access-token" > .env
```
**📌 Note:**
You can get your Hugging Face access token from ['https://huggingface.co/settings/tokens'](https://huggingface.co/settings/tokens)

#### 3. Install dependencies
```
pip install -r requirements.txt
```

#### 4. Launch the app
```
python gradio-dashboard.py
```

## Demo

<img src="https://github.com/Swuzz123/Book-Recommendation/raw/master/demo/demo_app.gif" width="1000">

## _**Feel free to use!**_
