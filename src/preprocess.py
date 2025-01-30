import pandas as pd
import torch
from transformers import BertTokenizer

# Load dataset
df = pd.read_csv(",,/data/tripadvisor_hotel_reviews.csv")

# Convert ratings into binary sentiment (positive: 4-5, negative: 1-3)
df['Sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize function
def tokenize_data(review):
    return tokenizer(review, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

df['Tokenized'] = df['Review'].apply(tokenize_data)

# Save preprocessed data
print(df.columns)
df.to_pickle("../data/preprocessed_tripadvisor.pkl")
print("Preprocessing complete. Saved tokenized data.")
