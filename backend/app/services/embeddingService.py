from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class EmbeddingService:
    def __init__(self, embedding_model=embedding_model):
        self.model = embedding_model

    def get_embedding(self, text):
        return self.model.embed_query(text)
    
    def get_embeddings(self, texts):
        return self.model.embed_documents(texts)