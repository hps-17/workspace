import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

##OPENROUTER_API_KEY="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"
load_dotenv()

# OpenRouter client (OpenAI-compatible API)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")  # Get free key from openrouter.ai
)

# ChromaDB setup (same as before)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def add_documents_to_db(collection_name, documents, metadatas=None, ids=None):
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    if metadatas is None: metadatas = [{"source": "doc"} for _ in documents]
    if ids is None: ids = [f"id{i}" for i in range(len(documents))]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(documents)} docs to {collection_name}")

def retrieve_context(collection_name, query, n_results=3):
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n".join(results["documents"][0])

def generate_response(prompt, context):
    """Fixed OpenRouter gpt-oss-20b call"""
    augmented_prompt = f"""Use this context to answer accurately:

CONTEXT:
{context}

QUESTION: {prompt}

Answer using only the provided context:"""
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # Correct model name [web:19]
        messages=[{"role": "user", "content": augmented_prompt}],
        temperature=0.1,
        extra_headers={
            "HTTP-Referer": "http://localhost:3000",  # Required
            "X-Title": "SAS-to-SQL RAG POC"  # Required
        }
    )
    return response.choices[0].message.content


# Usage
if __name__ == "__main__":
    sample_docs = [
        "PROC SQL; CREATE TABLE summary AS SELECT customer_id, SUM(sales) FROM transactions GROUP BY customer_id; QUIT;",
        "DATA step equivalent: SELECT * FROM table WHERE condition",
        "PROC MEANS = GROUP BY with aggregates in SQL"
    ]
    
    add_documents_to_db("sas_docs", sample_docs)
    
    context = retrieve_context("sas_docs", "SAS GROUP BY to SQL")
    response = generate_response("Convert SAS GROUP BY to SQL", context)
    
    print("Response:", response)
