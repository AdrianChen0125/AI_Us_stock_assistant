# Use official Airflow image
FROM apache/airflow:2.8.1

# Install basic libraries
USER root
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ python3-dev libffi-dev \
    libatlas-base-dev libblas-dev liblapack-dev gfortran \
    && apt-get clean

USER airflow

# Install Python packages
RUN pip install --upgrade pip && \
    pip install numpy hdbscan nltk

# Download NLTK resources separately
RUN python -m nltk.downloader punkt_tab stopwords wordnet

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prepare model path
RUN mkdir -p /opt/airflow/models && chmod -R 777 /opt/airflow/models
ENV MODEL_PATH /opt/airflow/models

# Download pretrained models
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', cache_dir='$MODEL_PATH'); \
AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', cache_dir='$MODEL_PATH')"

RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='$MODEL_PATH')"

RUN pip install dbt-core dbt-postgres

# Copy DAGs
COPY ./dags /opt/airflow/dags