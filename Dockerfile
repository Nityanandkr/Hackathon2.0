FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy project files
COPY . .

# Train the ML model during build
RUN python scripts/train_rf_model.py

# Expose the API port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
