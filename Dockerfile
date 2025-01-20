# Use a lightweight Python image as base
FROM python:3.12-slim

RUN useradd -d /app -m stats

USER stats

# Set working directory
WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy necessary files into the container
COPY . .

# Expose port where Flask will run
EXPOSE 5000

# Run Python application
CMD ["python", "app.py"]
