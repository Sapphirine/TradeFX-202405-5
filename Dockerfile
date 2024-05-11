# Use the official Python 3.9.12 image as base
FROM python:3.9.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Command to run the server.py script
CMD ["python", "server.py"]