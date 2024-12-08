# # Use your existing base image, for example, python:latest
# FROM python:latest

# # Set working directory in the container
# WORKDIR /app

# # Copy existing code (optional step if needed)
# COPY . /app

# # Install Data Science libraries (you can add more as needed)
# RUN pip install --upgrade pip
# RUN pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab

# # Expose port for Jupyter
# # EXPOSE 8888

# # Run Jupyter Notebook on container start
# CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]


# Use the official Python image as the base image
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
CMD ["python", "data_visualisation.py"]