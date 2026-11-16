import os
from typing import List, Optional, Any
from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation wrapping FastEmbed's ONNX TextEmbedding.
    Runs locally on ONNX runtime with zero PyTorch or HuggingFace API dependencies.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        batch_size: int = 256,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE_PATH")
        self.threads = threads
        self.batch_size = batch_size
        self._kwargs = kwargs
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                **self._kwargs,
            )
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into plain Python floats."""
        embeddings_generator = self.model.embed(texts, batch_size=self.batch_size)
        return [[float(x) for x in emb] for emb in embeddings_generator]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into plain Python floats."""
        embeddings_generator = self.model.query_embed(text)
        first_emb = next(iter(embeddings_generator))
        return [float(x) for x in first_emb]


def get_embeddings() -> Embeddings:
    """
    Select embeddings implementation based on EMBEDDING_BACKEND environment variable.
    Options:
      - "fastembed" (default): FastEmbed ONNX runtime (no torch, lightweight)
      - "local-torch": Local HuggingFace sentence-transformers (torch-based, dev only)
    """
    backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower().strip()

    if backend == "local-torch":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return FastEmbedEmbeddings()
