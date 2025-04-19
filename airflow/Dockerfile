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
    pip install numpy && \
    pip install hdbscan && \
    pip install nltk && \
    python -m nltk.downloader punkt stopwords punkt_tab &&\
    pip install --no-cache-dir -r requirements.txt

# Copy DAGs
COPY ./dags /opt/airflow/dags
