from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "RaeesTahir/DistilBERT-Sentico-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to your project directory
model.save_pretrained("model_weights")
tokenizer.save_pretrained("tokenizer")
