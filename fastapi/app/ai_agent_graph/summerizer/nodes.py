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
    user_profile: dict  # {"age": str, "experience": str, "risk": str}
    translated_inputs: dict
    report: str

# 條件節點：判斷是否翻譯
def should_translate(state: AgentState) -> Literal["translate", "skip"]:
    return "translate" if state["language"].lower() in ["chinese", "zh", "\u4e2d\u6587"] else "skip"

# 翻譯節點：將輸入內容轉為中文
def translate_inputs(state: AgentState) -> AgentState:
    translated = {
        "economic_summary": translate_text(state["economic_summary"]),
        "sentiment_summary": translate_text(state["sentiment_summary"]),
        "stock_summary": translate_text(state["stock_summary"]),
    }
    return {**state, "translated_inputs": translated}

# 跳過翻譯（將原文放進 translated_inputs）
def skip_translation(state: AgentState) -> AgentState:
    return {
        **state,
        "translated_inputs": {
            "economic_summary": state["economic_summary"],
            "sentiment_summary": state["sentiment_summary"],
            "stock_summary": state["stock_summary"]
        }
    }

# LLM 節點：產生完整報告
def generate_report(state: AgentState) -> AgentState:
    prompt = build_prompt(
        lang=state["language"],
        user=state["user_profile"],
        econ=state["translated_inputs"]["economic_summary"],
        sentiment=state["translated_inputs"]["sentiment_summary"],
        stock=state["translated_inputs"]["stock_summary"]
    )

    messages = [
        SystemMessage(content="You are a senior investment strategist who synthesizes multi-source information."),
        HumanMessage(content=prompt)
    ]
    result = llm.invoke(messages)
    return {**state, "report": result.content}
