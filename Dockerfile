# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt ./

# Install dependencies using pip
RUN pip install -r requirements.txt

# Copy your application code
COPY . .

# Expose the port where Flask listens (usually 5000)
EXPOSE 5000

# Set the main script for the container
CMD ["python", "app.py"]  