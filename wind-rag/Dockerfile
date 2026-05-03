FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 先安装Python依赖（利用Docker缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY src/ ./src/
COPY data/ ./data/
COPY .env.example ./
COPY README.md ./

# 预下载嵌入模型（构建时缓存，避免运行时下载）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('shibing624/text2vec-base-chinese')"

# 构建索引（如果没有预构建的索引）
RUN cd /app && python src/build_index.py || echo "索引构建跳过（可能已存在）"

# 暴露Gradio端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# 启动Gradio Web界面
CMD ["python", "src/gradio_app.py"]
