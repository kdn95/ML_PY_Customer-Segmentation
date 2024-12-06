# Use your existing base image, for example, python:latest
FROM python:latest

# Set working directory in the container
WORKDIR /app

# Copy existing code (optional step if needed)
COPY . /app

# Install Data Science libraries (you can add more as needed)
RUN pip install --upgrade pip
RUN pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab

# Expose port for Jupyter
EXPOSE 8888

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]
