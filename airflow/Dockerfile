# Use official Airflow image
FROM apache/airflow:2.8.1

# Use root for installing system packages
USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && apt-get clean

# Switch to airflow user
USER airflow

# Install required Python packages and download NLTK data
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install numpy hdbscan nltk && \
    python -m nltk.downloader punkt stopwords &&\
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /opt/airflow/models && chmod -R 777 /opt/airflow/models

RUN python3 -m nltk.downloader punkt

ENV MODEL_PATH=/opt/airflow/models
# sentiment model 

RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', cache_dir='$MODEL_PATH'); \
AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', cache_dir='$MODEL_PATH')"

# berttopic model
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='$MODEL_PATH')"


# Copy DAGs
COPY ./dags /opt/airflow/dags
