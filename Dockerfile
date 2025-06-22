FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "1_Home_Page.py", "--server.port", "8501", "--server.headless", "true"]
