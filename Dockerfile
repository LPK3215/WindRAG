FROM python:3.11-slim

# 主线（src/pure_rag/ + src/rag_cli.py）只用 Python 标准库，因此这里不装任何第三方包，
# 也不使用 requirements.txt（那是遗留研究版才需要的重依赖，见 AGENTS.md 项目自身需求第 3 条）。
WORKDIR /app

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY README.md ./
COPY config.env.example ./

# 构建索引：首次启动即有可检索的索引，不需要用户自己先跑一次 build
RUN python src/rag_cli.py build

EXPOSE 8000

# 健康检查打的是 /api/stats（对话页与静态资源不受影响，只确认服务真的在响应）
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/stats')" || exit 1

# 必须绑 0.0.0.0：容器内 127.0.0.1 无法被宿主机端口映射访问
CMD ["python", "src/rag_cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
