import os
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

# Initialize embedding model only
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.getenv("MODEL_PATH", "/models"))

async def retrieve_relevant_context(question: str, top_k: int, db):
    # 1. Encode the question into a vector
    vector_list = model.encode([question])[0].tolist()
    vector = f"[{', '.join(str(x) for x in vector_list)}]"
    print(vector)
    # 2. Retrieve top-k similar chunks with metadata
    query = """
    SELECT 
        c.chunk_text, 
        a.url
    FROM rag_docs.news_chunks c
    JOIN rag_docs.news_articles a 
        ON c.article_id = a.id
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
    """
    await db.execute("SET enable_seqscan = off")
    rows = await db.fetch(query, vector, top_k)

    chunks = [{"chunk_text": row['chunk_text'], "url": row['url']} for row in rows]

    return {
        "question": question,
        "context_used": chunks
    }