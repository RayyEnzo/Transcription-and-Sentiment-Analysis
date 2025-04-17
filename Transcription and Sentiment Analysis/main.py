import streamlit as st  # Imports the Streamlit library, which is used for creating web-based applications.
import os  # Imports the OS module for various operating system-related operations.
from transcribe import transcribe_audio  # Imports the transcribe_audio function from the transcribe module.
from sentiment import perform_sentiment_analysis, display_sentiment_analysis_scores, display_histogram, display_scatter_plot  # Imports functions and components from the sentiment module.
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Imports components from the Hugging Face Transformers library for BERT model usage.


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Sets an environment variable to disable parallelism in the Tokenizers library.

st.title("Sentico - Sentiment Analysis Toolkit")

# Upload audio file with Streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

transcription = None  # Initialize the transcription variable

if st.sidebar.button("Transcribe Audio"):
    if audio_file is not None:
        st.sidebar.success("Transcribing Audio")

        # Save the uploaded audio file to a temporary file
        with open("temp_audio_file.wav", "wb") as temp_file:
            temp_file.write(audio_file.read())

        # Transcribe the audio using Whisper
        transcription = transcribe_audio("temp_audio_file.wav")
        st.sidebar.success("Transcription Complete")
        st.markdown(transcription["text"])

    else:
        st.sidebar.error("Please upload an audio file")

# Initialize scores outside the conditional block
scores = None

# Load the model and tokenizer from the local directory
model_directory = "model_weights"
tokenizer_directory = "tokenizer"

# Set the cache directory to your project directory
cache_dir = model_directory

model = AutoModelForSequenceClassification.from_pretrained(model_directory, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory, cache_dir=cache_dir)

# Analyze sentiment of the transcribed text
if transcription:
    text_to_analyze = transcription["text"]  # Get the transcribed text
    max_seq_length = 512  # Define your desired max sequence length

    # Perform sentiment analysis using the model object
    scores, custom_labels = perform_sentiment_analysis(text_to_analyze, model, tokenizer, max_seq_length)
    display_sentiment_analysis_scores(scores, custom_labels)

    # After displaying the sentiment bar chart, you can call the histogram and scatter plot functions if desired
    if scores.dim() > 1:
        scores = scores.squeeze().detach().numpy()  # Remove extra dimensions if present
        display_histogram(scores, custom_labels)
        display_scatter_plot(range(len(custom_labels)), scores, custom_labels)
