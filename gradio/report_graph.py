from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from my_tools import *  # import your own steps

def get_graph():
    graph = StateGraph()

    graph.add_node("profile", RunnableLambda(get_user_profile))
    graph.add_node("eco", RunnableLambda(fetch_economic))
    graph.add_node("sentiment", RunnableLambda(fetch_sentiment))
    graph.add_node("enrich_news", RunnableLambda(enrich_with_news))
    graph.add_node("interest_news", RunnableLambda(fetch_user_interest_news))
    graph.add_node("final", RunnableLambda(generate_final_gpt_report))

    graph.set_entry_point("profile")
    graph.add_edge("profile", "eco")
    graph.add_edge("eco", "sentiment")
    graph.add_edge("sentiment", "enrich_news")
    graph.add_edge("enrich_news", "interest_news")
    graph.add_edge("interest_news", "final")
    graph.set_finish_point("final")

    return graph.compile()
