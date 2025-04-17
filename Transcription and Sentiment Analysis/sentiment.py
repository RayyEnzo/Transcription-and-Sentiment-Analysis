import streamlit as st  # Imports the Streamlit library, which is used for creating web-based applications.
import matplotlib.pyplot as plt  # Imports Matplotlib for creating visualizations.
import torch  # Imports PyTorch for tensor operations.


def perform_sentiment_analysis(transcribed_text, model, tokenizer, max_seq_length):
    # Tokenize the text and perform sentiment analysis
    tokenized_text = tokenizer(transcribed_text, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    input_ids = tokenized_text["input_ids"]

    # Split the input into chunks of size max_seq_length
    input_chunked = torch.split(input_ids, max_seq_length, dim=1)

    # Initialize a list to store sentiment scores
    sentiment_scores = []

    for chunk in input_chunked:
        sentiment_result = model(input_ids=chunk)
        sentiment_scores.append(sentiment_result.logits)

    # Combine scores from different chunks if necessary
    if len(sentiment_scores) > 1:
        scores = torch.cat(sentiment_scores).mean(dim=0)
    else:
        scores = sentiment_scores[0]

    # Custom sentiment labels
    custom_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    # Ensure that the scores sum to 100 (as percentages)
    total_score = scores.sum().item()
    if total_score > 0:
        scores = (scores / total_score) * 100

    return scores, custom_labels


def display_sentiment_analysis_scores(scores, custom_labels):
    # Create a simple bar chart using Matplotlib
    fig, ax = plt.subplots()

    if scores.dim() > 1:
        scores = scores.squeeze().detach().numpy()  # Remove extra dimensions if present
        ax.bar(custom_labels, scores, color='C0')
        ax.set_title("Sentiment Analysis Scores")
        ax.set_ylabel("Class probability (%)")
    else:
        st.warning("Unable to create a bar chart. The scores tensor has insufficient data.")

    # Display the chart using Streamlit
    st.pyplot(fig)

    # Display the sentiment label
    sentiment_label = custom_labels[scores.argmax().item()]
    st.sidebar.success("Sentiment Analysis Complete")
    st.markdown(f"Sentiment: {sentiment_label}")


def display_histogram(scores, custom_labels):
    # Create a histogram
    fig, ax = plt.subplots()
    ax.hist(scores, bins=10, color='C4', edgecolor='black')
    ax.set_title("Sentiment Analysis Scores (Histogram)")
    ax.set_xlabel("Class probability (%)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def display_scatter_plot(x_values, y_values, custom_labels):
    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, color='C3', marker='o')
    ax.set_title("Sentiment Analysis Scores (Scatter Plot)")
    ax.set_ylabel("Class probability (%)")
    ax.set_xlabel("Sentiment Class")
    ax.set_xticklabels(custom_labels, rotation=45)
    st.pyplot(fig)
