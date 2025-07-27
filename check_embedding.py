import joblib
import numpy as np

tokenizer_bilstm = joblib.load("models/tokenizer_bilstm_y.joblib")
embedding_matrix = np.load("models/embedding_matrix_y.npy")

print("Embedding matrix shape:", embedding_matrix.shape)
print("Tokenizer vocab size:", len(tokenizer_bilstm.word_index))
print("Contoh 10 kata pertama:")
for w, i in list(tokenizer_bilstm.word_index.items())[:10]:
    print(f"{w} : {i}")
