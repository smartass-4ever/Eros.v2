# Bella, live on Fly.io — she reads the web + thinks, and serves the "watch her think" surface.
FROM python:3.11-slim

WORKDIR /app

# build deps for any wheels that need compiling
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8080

EXPOSE 8080

# one process: static server for the surface (thread) + her cognition loop writing state.json
CMD ["python", "bella_live.py"]
