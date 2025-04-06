import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SHLRecommender:
    def __init__(self, catalog_path="catalog_data.csv"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.catalog = pd.read_csv(catalog_path)
        self.catalog["embedding"] = self.catalog["Tags"].apply(lambda x: self.model.encode(x, convert_to_tensor=True))

    def recommend(self, query, top_k=10):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        self.catalog["score"] = self.catalog["embedding"].apply(lambda emb: float(util.pytorch_cos_sim(query_embedding, emb)))
        top_results = self.catalog.sort_values("score", ascending=False).head(top_k)
        return top_results[["Assessment Name", "URL", "Remote Support", "Adaptive Support", "Duration", "Type", "score"]]
