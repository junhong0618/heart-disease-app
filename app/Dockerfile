# Use official Python base image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all project files to container
COPY app/ /app/

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit’s default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "eda.py", "--server.port=8501", "--server.address=0.0.0.0"]