FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only specific folders (see section 2 below)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./GRU ./GRU
COPY ./XGB ./XGB
COPY ./inference ./inference
COPY ./app.py ./app.py
COPY ./main.py ./main.py

EXPOSE 8501

# run the main python file, then start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8501"]

