from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('./model_cache/all-MiniLM-L6-v2')
print("Model saved to ./model_cache/all-MiniLM-L6-v2")