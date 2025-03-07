FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs models results visualizations

# Command to run when container starts
ENTRYPOINT ["python", "main.py"]