FROM python:3.12-slim

# uv'yi resmi image'ından kopyalayarak kuruyoruz — pip ile kurmaktan daha hızlı
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Önce sadece bağımlılık dosyalarını kopyalayıp yükle — bu sayede kod
# değiştiğinde bağımlılık katmanı Docker cache'inden yeniden kullanılır,
# her build'de yeniden indirme yapılmaz.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Şimdi geri kalan kodu kopyala
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/algorithm_arena/app/dashboard.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]