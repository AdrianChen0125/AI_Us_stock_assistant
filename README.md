# Overall Structure 
<img width="798" alt="Screenshot 2025-04-19 at 11 33 58 am copy" src="https://github.com/user-attachments/assets/e74711a2-1ca5-4db1-bb8d-38f283f45085" />


# AI US Stock investment assitant 

This is an end-to-end AI-driven platform that integrates US stock sentiment, macroeconomic indicators, and retrieval-augmented generation (RAG) to provide personalised financial insights. It combines data engineering, NLP, and interactive interfaces for users.

---

## Overview

- Collects social commentary and economic data related to US markets
- Applies NLP models to classify sentiment and summarise discussions
- Generates BI-style reports and enables real-time interaction via a conversational AI agent
- Supports agent workflow traceability to enhance transparency and explainability

---

## Architecture

### Data Layer

- Sources: YouTube, Reddit, and document uploads
- data stored in PostgreSQL  (Amazon RDS)
- Sentiment classification and embedding via Hugging Face models

### ELT Layer

- Apache Airflow handles batch ETL pipelines and model inference
- dbt transforms data across raw, processed, and production layers

### Backend

- FastAPI serves the core API and model endpoints
- LangGraph manages multi-step agent workflows for decision traceability
- MLflow stores model versions, metrics, and artefacts in S3
- RAG-based AI assistant performs document retrieval and LLM-based response generation

### Frontend

- Built using Gradio
- Provides:
  - Interactive chatbot powered by LangGraph + RAG
  - BI report viewer for summarised sentiment and economic analysis
  - Agent workflow viewer for debugging and traceability

---

## Features

- Social sentiment analysis for stock-related content
- Economic data integration (e.g., inflation, rates, GDP)
- Retrieval-augmented AI assistant for market Q&A
- BI report generation per user or query scope
- Agent workflow trace with LangGraph for explainable AI logic
- Fully containerised deployment

---

## Technology Stack

| Tool / Framework        | Purpose                                  |
|-------------------------|------------------------------------------|
| Apache Airflow          | ETL orchestration                        |
| Hugging Face            | NLP and sentiment modelling              |
| PostgreSQL (Amazon RDS) | Structured data storage                  |
| dbt                     | Data transformation                      |
| FastAPI                 | Backend services                         |
| LangGraph               | Agent orchestration                      |
| MLflow + S3             | Model tracking and storage               |
| RAG (LLM + Retrieval)   | AI assistant with contextual answers     |
| Gradio                  | Frontend UI                              |
| Docker                  | Deployment and environment management    |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
    
