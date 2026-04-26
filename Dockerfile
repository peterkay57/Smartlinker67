# Use the official Crawl4AI image which includes Chromium and Playwright
FROM unclecode/crawl4ai:latest

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY app.py .
COPY static/ ./static/

# Install additional dependencies (FastAPI app requirements)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["python", "app.py"]
