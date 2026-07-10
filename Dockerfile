FROM python:3.12-slim

WORKDIR /app

# Upgrade pip only — all packages have prebuilt wheels, apt-get skipped (no network in build)
RUN pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]