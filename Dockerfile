FROM python:3.8-slim-buster

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on (adjust if needed)
EXPOSE 5000

# Use exec form instead of shell form
CMD ["python", "app.py"]