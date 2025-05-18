import os
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 初始化嵌入模型和 LLM
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=os.getenv("MODEL_PATH", "/models"))
llm = ChatOpenAI(model="gpt-3.5-turbo")

async def process_rag_question(question: str, top_k: int, db):
    # 1. 轉換問題成向量
    vector_list = model.encode([question])[0].tolist()
    vector = f"[{', '.join(str(x) for x in vector_list)}]"

    # 2. 相似查詢
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

    # 3. 組 prompt 並用 LangChain 調用 GPT（支援 MLflow autolog）
    context = "\n\n".join(chunks)
    prompt = f"根據以下資料回答問題：\n\n{context}\n\n問題：{question}"

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "question": question,
        "answer": response.content.strip(),
        "context_used": chunks
    }