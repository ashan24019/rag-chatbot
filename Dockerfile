FROM python:3.11-slim

WORKDIR /app

# Stops Python writing .pyc files — not useful in containers
ENV PYTHONDONTWRITEBYTECODE=1

# Logs appear immediately instead of being buffered
ENV PYTHONUNBUFFERED=1

# Copy requirements FIRST — Docker caches this layer
# If only your app code changes, Docker skips pip install on rebuild
# This saves 2-3 minutes on every rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy app code — this layer rebuilds on every code change
COPY . .

CMD ["python", "main.py"]