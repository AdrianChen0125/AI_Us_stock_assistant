from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# 初始化模型與 GPT
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.getenv("MODEL_PATH", "/models"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_rag_question(question: str, top_k: int, db):
    # 1. Encode question to vector
    vector_list = model.encode([question])[0].tolist()
    vector = f"[{', '.join(str(x) for x in vector_list)}]"

    # 2. 查詢相似內容
    query = """
        SELECT chunk_text
        FROM rag_docs.news_chunks
        ORDER BY embedding <-> $1
        LIMIT $2
    """
    rows = await db.fetch(query, vector, top_k)
    chunks = [row['chunk_text'] for row in rows]

    if not chunks:
        return {
            "question": question,
            "answer": "找不到相關資料，請換個問題試試。",
            "context_used": []
        }

    # 3. 呼叫 GPT 組回答
    context = "\n\n".join(chunks)
    prompt = f"根據以下資料回答問題：\n\n{context}\n\n問題：{question}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    return {
        "question": question,
        "answer": answer,
        "context_used": chunks
    }