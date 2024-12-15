# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory
WORKDIR /prod

# Copy the requirements file
COPY requirements.txt /prod/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY brainmap brainmap

# Set environment variables
ENV PORT=8000

# Command to run the application
CMD ["uvicorn", "brainmap.api.fast:app", "--host", "0.0.0.0", "--port", "$PORT"]
