import pandas as pd
from omegaconf import DictConfig
from loguru import logger
from typing import List, Optional

from reqap.classes.observable_event import ObservableEvent
from reqap.retrieval.splade.sparse_retrieval import SparseRetrieval
from reqap.retrieval.splade.models import Splade
from reqap.retrieval.splade.index_construction import CollectionDataset
from reqap.retrieval.crossencoder.crossencoder_module import CrossEncoder
from reqap.retrieval.retrieval_pattern import RetrievalPattern


class Retrieval():
    def __init__(self, config: DictConfig, obs_events_csv_path: str, splade_index_path: str, dense_index_path: Optional[str] = None):
        self.config = config
        self.ce_config = config.crossencoder
        self.splade_config = config.splade
        self.dense_config = config.get("dense", {})
        self.hybrid_config = config.get("hybrid", {})
        
        splade_model = Splade(self.splade_config.splade_model_type_or_path, agg="max")
        collection = CollectionDataset(data_path=obs_events_csv_path)
        self.observable_events_df = collection.to_df()
        self.event_data_df = pd.DataFrame(self.observable_events_df["event_data"].tolist())
        self.sparse_retrieval = SparseRetrieval(
            splade_config=self.splade_config,
            model=splade_model,
            collection=collection,
            dim_voc=splade_model.output_dim,
            splade_index_path=splade_index_path
        )
        self.splade_involve_model = self.splade_config.get("splade_involve_model", True)
        
        # Initialize Dense Retriever and Hybrid Retrieval (if enabled)
        self.use_hybrid = self.hybrid_config.get("enabled", False)
        self.dense_retrieval = None
        self.hybrid_retrieval = None
        
        if self.use_hybrid:
            if dense_index_path is None:
                logger.warning("Hybrid retrieval enabled but dense_index_path not provided. Falling back to SPLADE only.")
                self.use_hybrid = False
            else:
                try:
                    from reqap.retrieval.dense.dense_retrieval import DenseRetrieval
                    from reqap.retrieval.hybrid_retrieval import HybridRetrieval
                    
                    self.dense_retrieval = DenseRetrieval(
                        dense_config=self.dense_config,
                        collection=collection,
                        dense_index_path=dense_index_path
                    )
                    self.hybrid_retrieval = HybridRetrieval(
                        sparse_retrieval=self.sparse_retrieval,
                        dense_retrieval=self.dense_retrieval,
                        fusion_strategy=self.hybrid_config.get("fusion_strategy", "rrf"),
                        fusion_params=self.hybrid_config.get("fusion_params", {})
                    )
                    logger.info(f"Hybrid retrieval enabled with fusion strategy: {self.hybrid_config.get('fusion_strategy', 'rrf')}")
                except Exception as e:
                    logger.error(f"Failed to initialize dense retriever: {e}. Falling back to SPLADE only.")
                    self.use_hybrid = False
        
        self.crossencoder = CrossEncoder(config=config, ce_config=self.ce_config)
        self.cache = dict()

    def retrieve(self, query: str, ordered: bool=False) -> List[ObservableEvent]:
        """
        Function to implement RETRIEVE function.
        Involves SPLADE for high-recall first-stage retrieval,
        and a cross-encoder classification model.

        By setting ordered = True, we can use it within RAG model.
        """
        # try to access cache
        if query in self.cache:
            return self.cache[query]
        
        """
        Step 1: Hybrid Retrieval or Sparse Retrieval - retrieve candidates via SPLADE and/or Dense Retriever.
        """
        if self.use_hybrid:
            # Use hybrid retrieval
            top_k_sparse = self.hybrid_config.get("top_k_sparse", 1000)
            top_k_dense = self.hybrid_config.get("top_k_dense", 1000)
            sparse_threshold = self.splade_config.get("splade_threshold", 0)
            dense_threshold = self.dense_config.get("dense_threshold", 0.0)
            final_top_k = self.hybrid_config.get("final_top_k", 0)  # 0 = return all
            
            candidates = self.hybrid_retrieval.retrieve(
                query=query,
                top_k_sparse=top_k_sparse,
                top_k_dense=top_k_dense,
                sparse_threshold=sparse_threshold,
                dense_threshold=dense_threshold,
                final_top_k=final_top_k
            )
            
            # Extract scores for later use
            event_to_splade_score = {}
            event_to_dense_score = {}
            event_to_hybrid_score = {}
            
            for d in candidates:
                # Handle both formats: {"data": {...}} and direct format
                if "data" in d:
                    doc_id = int(d["data"]["id"])
                else:
                    doc_id = int(d.get("id", 0))
                
                event_to_hybrid_score[doc_id] = d.get("hybrid_score", d.get("score", 0.0))
                # Try to extract original scores if available
                if "sparse_rank" in d or "sparse_score" in d:
                    event_to_splade_score[doc_id] = d.get("sparse_score", d.get("score", 0.0))
                if "dense_rank" in d or "dense_score" in d:
                    event_to_dense_score[doc_id] = d.get("dense_score", 0.0)
            
            logger.debug(f"Hybrid retrieval returned {len(candidates)} candidates")
        else:
            # Fall back to SPLADE only
            threshold = self.splade_config.get("splade_threshold", 0)
            candidates, _ = self.sparse_retrieval.retrieve(
                query,
                involve_model=self.splade_involve_model,  # whether to run model to expand query or not
                top_k=0,
                threshold=threshold
            )
            logger.debug(f"SPLADE threshold in use: {threshold}")
            logger.debug(f"{len(candidates)} events after SPLADE retrieval.")
            event_to_splade_score = {int(d["data"]["id"]): d["score"] for d in candidates}
            event_to_dense_score = {}
            event_to_hybrid_score = {}

        # avoids computing full CE result for extremely large outputs in RAG
        if ordered:
            candidates = candidates[:10000]

        # for ablation study: skip cross-encoder completely
        if self.splade_config.get("splade_only", False):
            obs_events = []
            for d in candidates:
                oe = ObservableEvent.from_dict(d["data"])
                splade_score = event_to_splade_score.get(int(oe.id), "None")
                dense_score = event_to_dense_score.get(int(oe.id), None)
                hybrid_score = event_to_hybrid_score.get(int(oe.id), None)
                oe.set_retrieval_result(
                    derived_via="RETRIEVAL_ONLY",
                    splade_score=splade_score if splade_score != "None" else None,
                    dense_score=dense_score,
                    hybrid_score=hybrid_score
                )
                obs_events.append(oe)
            return obs_events
        
        """
        Step 2: Pattern Detection - identify candidate positive and negative patterns.
        """
        # Convert candidates to unified format for pattern detection (expects {"id": ..., "data": {...}})
        candidates_for_pattern = []
        for d in candidates:
            if "data" in d:
                candidates_for_pattern.append(d)
            else:
                # Convert to expected format
                candidates_for_pattern.append({"id": d.get("id"), "data": d, "score": d.get("score", 0.0)})
        
        if self.ce_config.retrieval_pattern.apply:
            positive_patterns = RetrievalPattern.identify_candidate_positive_patterns(
                retrieval_result=candidates_for_pattern,
                min_events_matched=self.ce_config.retrieval_pattern.min_events_matched_inference
            )
            negative_patterns = RetrievalPattern.identify_candidate_negative_patterns(candidates_for_pattern)
        else:
            positive_patterns = list()
            negative_patterns = list()
        
        # Convert to ObservableEvent objects
        candidate_obs_events = []
        for d in candidates:
            if "data" in d:
                candidate_obs_events.append(ObservableEvent.from_dict(d["data"]))
            else:
                candidate_obs_events.append(ObservableEvent.from_dict(d))
        candidate_obs_events_dict = {int(e.id): e for e in candidate_obs_events}

        """
        Step 3: Pattern Classification - classify candidate patterns into (0) irrelevant, (1) partially relevant, and (2) relevant.
        """
        patterns = positive_patterns + negative_patterns
        if self.ce_config.retrieval_pattern.apply:
            scored_patterns = self.crossencoder.run_for_patterns(query, patterns)
            logger.debug(f"Scored patterns: {scored_patterns}")
            if self.ce_config.get("unified_negative_patterns", False):
                accepted_positive_patterns = [(c["pattern"], c["probabilities"]) for c in scored_patterns if c["class"] == 2]
                accepted_negative_patterns = [(c["pattern"]) for c in scored_patterns if c["class"] == 0]
            else:
                accepted_positive_patterns = [(c["pattern"], c["probabilities"]) for c in scored_patterns[:len(positive_patterns)] if c["class"] == 2]
                accepted_negative_patterns = [(c["pattern"]) for c in scored_patterns[len(positive_patterns):] if c["class"] == 0]
            logger.debug(f"Found {len(accepted_positive_patterns)} positive patterns for query=`{query}`: {accepted_positive_patterns}.")
            logger.debug(f"Found {len(accepted_negative_patterns)} negative patterns for query=`{query}`: {accepted_negative_patterns}.")

            
            # Apply positive patterns
            positive_events_dict = dict()
            for pattern, pattern_probs in accepted_positive_patterns:
                filtered_df = RetrievalPattern.apply_positive_pattern(self.observable_events_df, self.event_data_df, pattern)
                
                # add positive candidates from filtered df
                positive_events = ObservableEvent.from_df(filtered_df)
                for oe in positive_events:
                    splade_score = event_to_splade_score.get(int(oe.id), "PATTERN_ONLY")
                    dense_score = event_to_dense_score.get(int(oe.id), None)
                    hybrid_score = event_to_hybrid_score.get(int(oe.id), None)
                    oe.set_retrieval_result(
                        derived_via=pattern,
                        splade_score=splade_score if splade_score != "PATTERN_ONLY" else None,
                        dense_score=dense_score,
                        hybrid_score=hybrid_score,
                        ce_scores=pattern_probs
                    )
                    positive_events_dict[int(oe.id)] = oe  # this helps to avoid duplicates due to same event matched by multiple patterns
            positive_events = [oe for _, oe in positive_events_dict.items()]


            # Apply negative patterns
            # => IMPORTANT: positive patterns are prioritized over conflicting negative patterns
            candidate_obs_events = [ev for ev_id, ev in candidate_obs_events_dict.items() if not int(ev_id) in positive_events_dict]
            num_events_before_negative_patterns = len(candidate_obs_events)
            for pattern in accepted_negative_patterns:
                candidate_obs_events = RetrievalPattern.apply_negative_pattern(candidate_obs_events, pattern)
            logger.debug(f"Dropped {num_events_before_negative_patterns - len(candidate_obs_events)} events with negative patterns.")
            logger.debug(f"{len(candidate_obs_events)} candidate events remaining after SPLADE retrieval and pattern matching.")
        else:
            positive_events = list()

        """
        Step 4: Event Classification - classify remaining events into (0) irrelevant, and (1) relevant.
        """
        if candidate_obs_events:
            scored_candidates = self.crossencoder.run_for_events(query, candidate_obs_events, ordered=ordered)
            for c in scored_candidates:
                if c["class"]:
                    oe = c["obs_event"]
                    splade_score = event_to_splade_score.get(int(oe.id), None)
                    dense_score = event_to_dense_score.get(int(oe.id), None)
                    hybrid_score = event_to_hybrid_score.get(int(oe.id), None)
                    oe.set_retrieval_result(
                        derived_via="EVENT",
                        splade_score=splade_score,
                        dense_score=dense_score,
                        hybrid_score=hybrid_score,
                        ce_scores=c["probabilities"]
                    )
                    positive_events.append(oe)
        logger.debug(f"{len(positive_events)} positive events after cross-encoder classification.")

        """
        Step 5: Event Deduplication - deduplicate events that express same information.
        Done in `create_computed_events` function, after predicting temporal information.
        """

        # store in cache
        self.cache[query] = positive_events
        return positive_events
    
    def load(self):
        self.crossencoder.load_models()

    def clear_cache(self):
        self.cache = dict()