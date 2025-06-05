# nodes.py
from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from .prompt import build_prompt
from services.translation_service import translate_text

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

class AgentState(TypedDict):
    language: str
    economic_summary: str
    sentiment_summary: str
    stock_summary: str
    user_profile: dict 
    report: str

# LLM 節點：產生完整報告
def generate_report(state: AgentState) -> AgentState:
    prompt = build_prompt(
        lang=state["language"],
        user=state["user_profile"],
        econ=state["economic_summary"],
        sentiment=state["sentiment_summary"],
        stock=state["stock_summary"]
    )

    messages = [
        SystemMessage(content="You are a senior investment strategist who synthesizes multi-source information."),
        HumanMessage(content=prompt)
    ]
    result = llm.invoke(messages)
    return {**state, "report": result.content}

# 條件節點：判斷是否翻譯
def should_translate(state: AgentState) -> Literal["translate", "skip"]:
    return "translate" if state["language"].lower() in ["chinese", "zh", "\u4e2d\u6587"] else "skip"

def translate_service(state: AgentState) -> AgentState:
    target_lang = state["language"]
    original_report = state["report"]

    # 呼叫翻譯服務
    translated_report = translate_text(original_report, target_lang)
   

    # 回傳新的 state，覆蓋原本 report 為翻譯後版本
    return {**state, "report": translated_report}
