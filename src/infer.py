import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("/Users/sravanthikonduru/Desktop/tripadvisor/models/tripadvisor_optimized.pth", map_location=device))
model.to(device)
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Predict Function
def predict_sentiment(review):
    tokens = tokenizer(review, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids, attention_mask = tokens["input_ids"].to(device), tokens["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        prediction = torch.argmax(output.logits, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

# Test Inference
test_review = "The hotel was amazing, with great service and friendly staff!"
predicted_sentiment = predict_sentiment(test_review)

print(f"Review: {test_review}\nPredicted Sentiment: {predicted_sentiment}")

# Save Predictions
with open("predictions.txt", "w") as f:
    f.write(f"Review: {test_review}\nPredicted Sentiment: {predicted_sentiment}\n")
