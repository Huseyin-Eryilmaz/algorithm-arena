FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Önce sadece bağımlılıkları kur (proje kodu olmadan) — bu adım sadece
# pyproject.toml/uv.lock değiştiğinde yeniden çalışır, cache'ten faydalanır.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Şimdi geri kalan kodu kopyala
COPY . .

# Şimdi projenin kendisini kur (src/ artık mevcut)
RUN uv sync --frozen --no-dev

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/algorithm_arena/app/dashboard.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]