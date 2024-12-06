# Use Python for base image
FROM python:latest

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the container
COPY . .

# Specify the default command to run your application
CMD ["python", "app.py"]
