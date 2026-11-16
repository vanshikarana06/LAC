import sys
import os
import math
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.embeddings
from src.embeddings import FastEmbedEmbeddings, get_embeddings


class TestEmbeddingsModule(unittest.TestCase):
    def test_no_torch_or_transformers_import(self):
        """Prove that importing src.embeddings in a clean process does not import torch or transformers."""
        import subprocess

        cmd = [
            sys.executable,
            "-c",
            "import sys, src.embeddings; assert 'torch' not in sys.modules, 'torch imported'; assert 'transformers' not in sys.modules, 'transformers imported'",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode,
            0,
            f"Importing src.embeddings imported torch or transformers: {result.stderr}",
        )

    def test_fastembed_wrapper_with_monkeypatched_model(self):
        """Test output shapes, types, and document vs query methods with monkeypatched TextEmbedding."""
        # 384-dimensional normalized vector
        raw_vec = np.random.randn(384).astype(np.float32)
        norm_vec = raw_vec / np.linalg.norm(raw_vec)

        mock_instance = MagicMock()
        mock_instance.embed.return_value = [norm_vec.copy(), norm_vec.copy()]
        mock_instance.query_embed.return_value = iter([norm_vec.copy()])

        with patch("fastembed.TextEmbedding", return_value=mock_instance) as mock_cls:
            with patch.dict(os.environ, {"FASTEMBED_CACHE_PATH": "/custom/cache/path"}):
                embedder = FastEmbedEmbeddings()
                self.assertEqual(embedder.cache_dir, "/custom/cache/path")

                # Test embed_documents
                docs = ["First legal section.", "Second legal section."]
                doc_vecs = embedder.embed_documents(docs)

                self.assertIsInstance(doc_vecs, list)
                self.assertEqual(len(doc_vecs), 2)
                for vec in doc_vecs:
                    self.assertIsInstance(vec, list)
                    self.assertEqual(len(vec), 384)
                    self.assertTrue(all(isinstance(x, float) for x in vec))

                # Test embed_query
                query = "What is theft?"
                query_vec = embedder.embed_query(query)

                self.assertIsInstance(query_vec, list)
                self.assertEqual(len(query_vec), 384)
                self.assertTrue(all(isinstance(x, float) for x in query_vec))

                mock_cls.assert_called_once_with(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_dir="/custom/cache/path",
                    threads=None,
                )

    def test_l2_normalization_check(self):
        """Test that returned vectors are L2-normalized (norm approx 1.0)."""
        raw_vec = np.random.randn(384).astype(np.float32)
        norm_vec = raw_vec / np.linalg.norm(raw_vec)

        mock_instance = MagicMock()
        mock_instance.embed.return_value = [norm_vec]
        mock_instance.query_embed.return_value = iter([norm_vec])

        with patch("fastembed.TextEmbedding", return_value=mock_instance):
            embedder = FastEmbedEmbeddings()
            query_vec = embedder.embed_query("test query")
            doc_vecs = embedder.embed_documents(["test doc"])

            query_norm = math.sqrt(sum(x * x for x in query_vec))
            doc_norm = math.sqrt(sum(x * x for x in doc_vecs[0]))

            self.assertAlmostEqual(query_norm, 1.0, places=4, msg="Query vector must be L2-normalized")
            self.assertAlmostEqual(doc_norm, 1.0, places=4, msg="Document vector must be L2-normalized")

    def test_get_embeddings_backend_selection(self):
        """Test get_embeddings() selects backend according to EMBEDDING_BACKEND env."""
        # 1. Default backend -> FastEmbedEmbeddings
        with patch.dict(os.environ, {}, clear=True):
            emb = get_embeddings()
            self.assertIsInstance(emb, FastEmbedEmbeddings)

        # 2. Explicit 'fastembed' -> FastEmbedEmbeddings
        with patch.dict(os.environ, {"EMBEDDING_BACKEND": "fastembed"}):
            emb = get_embeddings()
            self.assertIsInstance(emb, FastEmbedEmbeddings)

        # 3. 'local-torch' -> HuggingFaceEmbeddings (lazy-imported)
        mock_hf_cls = MagicMock()
        with patch.dict(os.environ, {"EMBEDDING_BACKEND": "local-torch"}):
            with patch.dict("sys.modules", {"langchain_huggingface": MagicMock(HuggingFaceEmbeddings=mock_hf_cls)}):
                emb = get_embeddings()
                mock_hf_cls.assert_called_once_with(model_name="all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
