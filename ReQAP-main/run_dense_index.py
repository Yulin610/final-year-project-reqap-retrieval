"""
Script to build dense indices for hybrid retrieval.
"""
import sys
from loguru import logger
from omegaconf import DictConfig

from reqap.library.library import load_config, get_persona_names
from reqap.retrieval.dense.dense_retrieval import DenseRetrieval
from reqap.retrieval.splade.index_construction import CollectionDataset


def build_dense_index(config: DictConfig):
    """
    Build dense indices for all personas in train/dev/test splits.
    """
    benchmark_dir = config.benchmark.benchmark_dir
    dense_config = config.dense
    
    for split in ["train", "dev", "test"]:
        logger.info(f"Building dense index for {split} set...")
        persona_dir = f"{benchmark_dir}/{split}"
        personas = get_persona_names(persona_dir)
        
        for persona in personas:
            obs_events_csv_path = f"{persona_dir}/{persona}/{persona}_obs.csv"
            dense_index_path = f"{dense_config.dense_indices_dir}/{persona}.dense_index"
            
            logger.info(f"Processing {persona}...")
            collection = CollectionDataset(data_path=obs_events_csv_path)
            dense_retriever = DenseRetrieval(
                dense_config=dense_config,
                collection=collection
            )
            
            dense_retriever.build_index(
                output_path=dense_index_path,
                batch_size=dense_config.get("dense_index_batch_size", 32)
            )
            logger.info(f"Done with building dense index for {persona}")


def main():
    if len(sys.argv) < 2:
        raise Exception("Usage: python run_dense_index.py <CONFIG>")
    
    config_path = sys.argv[1]
    logger.debug(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    build_dense_index(config)
    logger.info("Done with building all dense indices!")


if __name__ == "__main__":
    main()






