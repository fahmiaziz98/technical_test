from langchain_huggingface import HuggingFaceEmbeddings


def embedding_pipeline(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    return embeddings

