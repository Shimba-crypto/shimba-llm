FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch numpy

COPY shimba.py .
COPY shimba_colab.ipynb .

EXPOSE 8000

CMD ["python", "shimba.py", "serve", "--model", "model.pth", "--port", "8000"]
