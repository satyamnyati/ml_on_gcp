FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and train inside the image (quick + reproducible)
COPY app/ /app/
RUN python train.py

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
