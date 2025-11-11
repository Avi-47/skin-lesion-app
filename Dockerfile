# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY backend /app/backend
COPY static /app/static
COPY index.html /app/
COPY README.md /app/

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Set environment variable for Flask
ENV FLASK_APP=backend/app.py

# Run the Flask app
CMD ["python", "backend/app.py"]