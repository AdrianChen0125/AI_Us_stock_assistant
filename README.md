# Overall Structure 
<img width="798" alt="Screenshot 2025-04-19 at 11 33 58 am copy" src="https://github.com/user-attachments/assets/e74711a2-1ca5-4db1-bb8d-38f283f45085" />

# 📊 US Stock Sentiment & Economic Intelligence Platform

> **Collect stock-related comments + real economic indicators → Generate personalized reports → Explore deeper insights via AI chatbot.**

---

## ✨ Project Overview

This project builds a full data-driven AI system that collects user-generated comments related to **US stocks** and combines them with **real-world economic indexes** to generate personalized financial reports for users.  
Users can also interact further with an **AI chatbot** to explore deeper financial insights.

---

## 🏗 Architecture

- **Data Sources**:
  - Scrapes comments from **YouTube**, **Reddit**, and document uploads.
  - Pulls economic data from external APIs (e.g., News API).

- **ETL Layer**:
  - **Apache Airflow** schedules and orchestrates ETL tasks.
  - **Hugging Face** models for text enrichment and sentiment analysis.
  - **PostgreSQL** (Amazon RDS) for raw, processed, and production data storage.
  - **dbt** models and transforms data for final production tables.

- **Implementation Layer**:
  - **Gradio** web interface for interactive user reports.
  - **LangChain** integrates database with OpenAI, Wikipedia, and News APIs.
  - **BI Report** generation using transformed datasets.

- **Deployment**:
  - Fully containerized with **Docker**.
  - Runs locally on **Mac** or any Docker-supported environment.

---

## 🚀 Key Features

- 🛠 Collects and classifies US stock-related comments.
- 📈 Merges sentiment data with real economic indicators.
- 📝 Generates customized financial reports.
- 🤖 AI chatbot allows users to explore stock sentiment, news, and market trends.
- 🔗 Supports integration with OpenAI, Wikipedia, News APIs.

---

## 🛠 Technologies Used

| Technology | Purpose |
|:----------|:--------|
| Apache Airflow | ETL orchestration |
| Hugging Face Transformers | NLP sentiment analysis |
| PostgreSQL (Amazon RDS) | Data storage |
| dbt | Data transformation |
| Gradio | Frontend web app |
| LangChain | AI integration |
| Docker | Containerization and deployment |

---

## 📦 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
    
