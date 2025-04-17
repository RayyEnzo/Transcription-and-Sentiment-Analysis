# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY .. /app

# Copy the model weights and tokenizer files into the container
COPY ../model_weights /app/model_weights
COPY ../tokenizer /app/tokenizer

# Install 'git' to be able to install Whisper
RUN apt-get update && apt-get install -y git

# Install ffmpeg for moviepy (used for audio processing)
RUN apt-get update && apt-get install -y ffmpeg

# Install Whisper from the Git repository
RUN pip3 install "git+https://github.com/openai/whisper.git"

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY .. .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "main.py"]
