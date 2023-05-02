import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# db_vectors = np.random.random((n, dimension)).astype('float32')

data = pd.read_csv('df_tofind.csv').dropna()
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

dimension = model.get_sentence_embedding_dimension()
number_of_vectors = len(data)
print(f"Models encoding size: {dimension}")

nlist = 5  # number of clusters
quantiser = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.IndexFlatIP())