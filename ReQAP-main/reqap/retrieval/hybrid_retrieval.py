"""
Hybrid retrieval combining SPLADE (sparse) and Dense Retriever.
Implements various fusion strategies for merging results.
"""
from typing import List, Dict
from loguru import logger
from collections import defaultdict

from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval


class HybridRetrieval:
    """
    Combines sparse (SPLADE) and dense retrieval results.
    """
    
    FUSION_STRATEGIES = ["rrf", "weighted_sum", "reciprocal_rank", "max"]
    
    def __init__(self, sparse_retrieval: SparseRetrieval, dense_retrieval: DenseRetrieval, 
                 fusion_strategy: str = "rrf", fusion_params: Dict = None):
        self.sparse_retrieval = sparse_retrieval
        self.dense_retrieval = dense_retrieval
        self.fusion_strategy = fusion_strategy
        self.fusion_params = fusion_params or {}
        
        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}. Choose from {self.FUSION_STRATEGIES}")
    
    def retrieve(self, query: str, top_k_sparse: int = 1000, top_k_dense: int = 1000,
                 sparse_threshold: float = 0.0, dense_threshold: float = 0.0,
                 final_top_k: int = 0) -> List[Dict]:
        """
        Perform hybrid retrieval by combining sparse and dense results.
        
        Args:
            query: Query string
            top_k_sparse: Top-K for sparse retrieval
            top_k_dense: Top-K for dense retrieval
            sparse_threshold: Threshold for sparse retrieval
            dense_threshold: Threshold for dense retrieval
            final_top_k: Final top-K after fusion (0 = return all)
            
        Returns:
            Merged and ranked results
        """
        # Parallel retrieval
        logger.debug(f"Performing hybrid retrieval with strategy: {self.fusion_strategy}")
        
        # Sparse retrieval
        sparse_results, _ = self.sparse_retrieval.retrieve(
            query,
            involve_model=True,
            top_k=top_k_sparse,
            threshold=sparse_threshold
        )
        logger.debug(f"SPLADE retrieved {len(sparse_results)} candidates")
        
        # Dense retrieval
        dense_results = self.dense_retrieval.retrieve(
            query,
            top_k=top_k_dense,
            threshold=dense_threshold
        )
        logger.debug(f"Dense retriever retrieved {len(dense_results)} candidates")
        
        # Merge results
        merged_results = self._fuse_results(sparse_results, dense_results)
        
        # Apply final top-K if specified
        if final_top_k > 0 and len(merged_results) > final_top_k:
            merged_results = merged_results[:final_top_k]
        
        logger.debug(f"Hybrid retrieval returned {len(merged_results)} final results")
        return merged_results
    
    def _fuse_results(self, sparse_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """
        Fuse sparse and dense retrieval results using specified strategy.
        """
        if self.fusion_strategy == "rrf":
            return self._reciprocal_rank_fusion(sparse_results, dense_results)
        elif self.fusion_strategy == "weighted_sum":
            return self._weighted_sum_fusion(sparse_results, dense_results)
        elif self.fusion_strategy == "reciprocal_rank":
            return self._reciprocal_rank_fusion(sparse_results, dense_results)
        elif self.fusion_strategy == "max":
            return self._max_fusion(sparse_results, dense_results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _reciprocal_rank_fusion(self, sparse_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).
        Score = sum(1 / (k + rank)) for each retrieval method
        
        Args:
            k: RRF constant (default: 60)
        """
        k = self.fusion_params.get("rrf_k", 60)
        
        # Build rank maps
        sparse_ranks = {r["id"]: rank + 1 for rank, r in enumerate(sparse_results)}
        dense_ranks = {r["id"]: rank + 1 for rank, r in enumerate(dense_results)}
        
        # Collect all unique document IDs
        all_doc_ids = set(sparse_ranks.keys()) | set(dense_ranks.keys())
        
        # Compute RRF scores
        doc_scores = defaultdict(float)
        doc_data = {}
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Sparse contribution
            if doc_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[doc_id])
                # Store document data from sparse results
                for r in sparse_results:
                    if r["id"] == doc_id:
                        doc_data[doc_id] = r
                        break
            
            # Dense contribution
            if doc_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[doc_id])
                # Store document data from dense results if not already stored
                if doc_id not in doc_data:
                    for r in dense_results:
                        if r["id"] == doc_id:
                            doc_data[doc_id] = r
                            break
            
            doc_scores[doc_id] = rrf_score
        
        # Build merged results
        # Ensure format matches SPLADE: {"id": ..., "score": ..., "data": {...}, "derivation": [...]}
        merged_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_data[doc_id].copy()
            # Extract data if it's already in the result, otherwise use the whole result as data
            if "data" not in result:
                # If result doesn't have "data" key, create it from the result itself
                data = {k: v for k, v in result.items() if k not in ["id", "score", "derivation"]}
                result = {
                    "id": result.get("id", doc_id),
                    "score": result.get("score", score),
                    "data": data,
                    "derivation": result.get("derivation", [])
                }
            result["score"] = score  # Update with hybrid score
            result["hybrid_score"] = score
            result["sparse_rank"] = sparse_ranks.get(doc_id, None)
            result["dense_rank"] = dense_ranks.get(doc_id, None)
            result["derivation"] = result.get("derivation", []) + [
                {"method": "hybrid_rrf", "score": score}
            ]
            merged_results.append(result)
        
        return merged_results
    
    def _weighted_sum_fusion(self, sparse_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """
        Weighted sum fusion.
        Score = alpha * normalized_sparse_score + (1 - alpha) * normalized_dense_score
        """
        alpha = self.fusion_params.get("alpha", 0.5)  # Weight for sparse retrieval
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            if not results:
                return {}
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            normalized = {}
            for r in results:
                norm_score = (r["score"] - min_score) / score_range
                normalized[r["id"]] = norm_score
            return normalized
        
        sparse_scores = normalize_scores(sparse_results)
        dense_scores = normalize_scores(dense_results)
        
        # Collect all document IDs
        all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        # Compute weighted scores
        doc_scores = {}
        doc_data = {}
        
        for doc_id in all_doc_ids:
            sparse_score = sparse_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            hybrid_score = alpha * sparse_score + (1 - alpha) * dense_score
            
            doc_scores[doc_id] = hybrid_score
            
            # Get document data
            if doc_id in sparse_scores:
                for r in sparse_results:
                    if r["id"] == doc_id:
                        doc_data[doc_id] = r
                        break
            elif doc_id in dense_scores:
                for r in dense_results:
                    if r["id"] == doc_id:
                        doc_data[doc_id] = r
                        break
        
        # Build merged results
        # Ensure format matches SPLADE: {"id": ..., "score": ..., "data": {...}, "derivation": [...]}
        merged_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_data[doc_id].copy()
            # Extract data if it's already in the result, otherwise use the whole result as data
            if "data" not in result:
                data = {k: v for k, v in result.items() if k not in ["id", "score", "derivation"]}
                result = {
                    "id": result.get("id", doc_id),
                    "score": result.get("score", score),
                    "data": data,
                    "derivation": result.get("derivation", [])
                }
            result["score"] = score  # Update with hybrid score
            result["hybrid_score"] = score
            result["sparse_score"] = sparse_scores.get(doc_id, 0.0)
            result["dense_score"] = dense_scores.get(doc_id, 0.0)
            result["derivation"] = result.get("derivation", []) + [
                {"method": "hybrid_weighted", "score": score, "alpha": alpha}
            ]
            merged_results.append(result)
        
        return merged_results
    
    def _max_fusion(self, sparse_results: List[Dict], dense_results: List[Dict]) -> List[Dict]:
        """
        Max fusion: take maximum of normalized scores.
        """
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return {}
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            normalized = {}
            for r in results:
                norm_score = (r["score"] - min_score) / score_range
                normalized[r["id"]] = norm_score
            return normalized
        
        sparse_scores = normalize_scores(sparse_results)
        dense_scores = normalize_scores(dense_results)
        
        # Take max
        all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        doc_scores = {}
        doc_data = {}
        
        for doc_id in all_doc_ids:
            sparse_score = sparse_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            doc_scores[doc_id] = max(sparse_score, dense_score)
            
            # Get document data (prefer the one with higher score)
            if sparse_score >= dense_score and doc_id in sparse_scores:
                for r in sparse_results:
                    if r["id"] == doc_id:
                        doc_data[doc_id] = r
                        break
            elif doc_id in dense_scores:
                for r in dense_results:
                    if r["id"] == doc_id:
                        doc_data[doc_id] = r
                        break
        
        # Build merged results
        # Ensure format matches SPLADE: {"id": ..., "score": ..., "data": {...}, "derivation": [...]}
        merged_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_data[doc_id].copy()
            # Extract data if it's already in the result, otherwise use the whole result as data
            if "data" not in result:
                data = {k: v for k, v in result.items() if k not in ["id", "score", "derivation"]}
                result = {
                    "id": result.get("id", doc_id),
                    "score": result.get("score", score),
                    "data": data,
                    "derivation": result.get("derivation", [])
                }
            result["score"] = score  # Update with hybrid score
            result["hybrid_score"] = score
            result["derivation"] = result.get("derivation", []) + [
                {"method": "hybrid_max", "score": score}
            ]
            merged_results.append(result)
        
        return merged_results






