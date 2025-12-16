FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (including data/ and pipelines/)
COPY . .

# Train models during build
# This ensures a fresh set of models is baked into the image
RUN python pipelines/train_simple.py

# Generate Data Validation Report during build
# This ensures valiation_report.html exists for the UI
RUN python pipelines/validate_data.py

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
