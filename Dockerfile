FROM python:3.9-slim

# ffmpegとffprobeをインストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswscale-extra \
    && rm -rf /var/lib/apt/lists/*

# アプリケーションの依存関係をインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# アプリケーションのコードをコピー
COPY . .

# ポートを公開
EXPOSE 8501

# アプリケーションを起動
CMD ["streamlit", "run", "app.py"] 