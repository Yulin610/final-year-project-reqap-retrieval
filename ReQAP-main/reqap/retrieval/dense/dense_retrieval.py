"""
Dense Retriever for hybrid retrieval with SPLADE.
Uses dense embeddings (e.g., sentence transformers) for semantic search.
"""
import os
import torch
import pickle
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from loguru import logger

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from reqap.library.library import move_model, normalize_tensor
from reqap.retrieval.splade.index_construction import CollectionDataset


class DenseRetrieval:
    """
    Dense retrieval using sentence transformers or other dense embedding models.
    """
    def __init__(self, dense_config: Dict, collection: CollectionDataset, dense_index_path: Optional[str] = None):
        self.dense_config = dense_config
        self.collection = collection
        self._doc_by_id: Dict[int, Dict] = {}
        for i in range(len(self.collection)):
            row = self.collection[i]["data"]
            self._doc_by_id[int(row["id"])] = row
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Load model
        model_type = dense_config.get("dense_model_type_or_path", "sentence-transformers/all-MiniLM-L6-v2")
        self.use_sentence_transformers = dense_config.get("use_sentence_transformers", True)
        
        if self.use_sentence_transformers:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is required for dense retrieval. Install it with: pip install sentence-transformers")
            logger.info(f"Loading SentenceTransformer model: {model_type}")
            self.model = SentenceTransformer(model_type, device=str(self.device))
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            logger.info(f"Loading AutoModel: {model_type}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_type)
            self.model = AutoModel.from_pretrained(model_type)
            move_model(self.model, self.device)
            self.embedding_dim = self.model.config.hidden_size
        
        # Load or build index
        if dense_index_path is not None:
            # Check if it's a directory (contains dense_index.faiss) or a file path
            if os.path.isdir(dense_index_path) and os.path.exists(os.path.join(dense_index_path, "dense_index.faiss")):
                self.load_index(dense_index_path)
            elif os.path.exists(dense_index_path):
                # If it's a file, try to load from parent directory
                parent_dir = os.path.dirname(dense_index_path)
                if os.path.exists(os.path.join(parent_dir, "dense_index.faiss")):
                    self.load_index(parent_dir)
                else:
                    logger.warning(f"Dense index not found at {dense_index_path}. Please build index first using build_index()")
                    self.index = None
                    self.doc_ids = None
            else:
                logger.warning(f"Dense index path does not exist: {dense_index_path}. Please build index first using build_index()")
                self.index = None
                self.doc_ids = None
        else:
            logger.warning("Dense index path not provided. Please build index first using build_index()")
            self.index = None
            self.doc_ids = None
    
    def build_index(self, output_path: str, batch_size: int = 32):
        """
        Build dense index for all events in the collection.
        """
        logger.info("Building dense index...")
        
        # Prepare texts for encoding
        texts = []
        doc_ids = []
        for i in range(len(self.collection)):
            doc_data = self.collection[i]["data"]
            doc_id = int(doc_data["id"])
            text = self._verbalize_event(doc_data)
            texts.append(text)
            doc_ids.append(doc_id)
        
        # Encode documents in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if self.use_sentence_transformers:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                batch_embeddings = self._encode_with_automodel(batch_texts)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Encoded {len(embeddings)} documents with dimension {embeddings.shape[1]}")
        
        # Build FAISS index
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for dense retrieval. Install it with: pip install faiss-cpu (or faiss-gpu)")
        
        self.embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity (normalized)
        self.index.add(embeddings.astype('float32'))
        self.doc_ids = np.array(doc_ids, dtype=np.int32)
        
        # Save index
        os.makedirs(output_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_path, "dense_index.faiss"))
        with open(os.path.join(output_path, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)
        logger.info(f"Saved dense index to {output_path}")
    
    def load_index(self, index_path: str):
        """
        Load pre-built dense index.
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required for dense retrieval. Install it with: pip install faiss-cpu (or faiss-gpu)")
        
        logger.info(f"Loading dense index from {index_path}")
        self.index = faiss.read_index(os.path.join(index_path, "dense_index.faiss"))
        with open(os.path.join(index_path, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = 1000, threshold: float = 0.0) -> List[Dict]:
        """
        Retrieve documents using dense embeddings.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieved documents with scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please build or load index first.")
        
        with torch.no_grad():
            # Encode query
            if self.use_sentence_transformers:
                query_embedding = self.model.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            else:
                query_embedding = self._encode_with_automodel([query])[0]
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search in index
            scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            scores = scores[0]
            indices = indices[0]
            
            # Filter by threshold and prepare results
            # Format to match SPLADE output format: {"id": ..., "score": ..., "derivation": [...], **doc_data}
            results = []
            for score, idx in zip(scores, indices):
                if score >= threshold:
                    doc_id = int(self.doc_ids[idx])
                    doc_data = self._doc_by_id.get(doc_id)
                    if doc_data is None:
                        continue
                    results.append({
                        "id": doc_id,
                        "score": float(score),
                        "derivation": [{"method": "dense", "score": float(score)}],
                        **doc_data
                    })
            
            # Sort by score (descending)
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            logger.debug(f"Dense retrieval returned {len(results)} results (top_k={top_k}, threshold={threshold})")
            return results
    
    def _encode_with_automodel(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using AutoModel (mean pooling).
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = normalize_tensor(embeddings)
        
        return embeddings.cpu().numpy()
    
    def _verbalize_event(self, event_data: Dict) -> str:
        """
        Convert event data to text for encoding.
        """
        event_type = event_data.get("event_type", "")
        event_data_dict = event_data.get("event_data", {})
        
        if isinstance(event_data_dict, str):
            import json
            try:
                event_data_dict = json.loads(event_data_dict)
            except:
                event_data_dict = {}
        
        # Build text representation
        parts = [f"Event type: {event_type}"]
        for key, value in event_data_dict.items():
            if isinstance(value, (int, float, str)):
                parts.append(f"{key}: {value}")
        
        return ". ".join(parts)

