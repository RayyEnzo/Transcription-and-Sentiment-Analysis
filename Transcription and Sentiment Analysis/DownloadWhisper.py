from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Define the model name for the Whisper ASR base model
model_name = "openai/whisper-base"
model = WhisperProcessor.from_pretrained(model_name)
tokenizer = WhisperForConditionalGeneration.from_pretrained(model_name)

# Save the model and tokenizer to your project directory
model.save_pretrained("whisper_tokenizer")
tokenizer.save_pretrained("whisper_model_weights")
