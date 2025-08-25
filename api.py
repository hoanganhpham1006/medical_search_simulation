from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from datasets import load_from_disk, load_dataset, concatenate_datasets
import os
import glob
import asyncio
from pathlib import Path
import logging
import config
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import subprocess
import requests
import httpx
import signal
import atexit
import sys
import re
from cache_utils import StartupDataCache
from faiss_index_manager import FaissIndexManager
import asyncio


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = config.DEBUG_MODE


# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    use_reranker: Optional[bool] = False  # Default to using reranker
    top_k: Optional[int] = None  # Number of search results to return (uses MAX_SEARCH_RESULTS if not specified)
    preview_char: Optional[int] = -1  # Number of preview characters to return (-1 to skip preview generation)


class SearchResult(BaseModel):
    url: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    preview: str = ""  # Text preview of the most relevant chunk


class SearchResponse(BaseModel):
    results: List[SearchResult]


class VisitRequest(BaseModel):
    url: str


class VisitResponse(BaseModel):
    url: str
    data: str
    status_code: int


# Global variables for models and data
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description="A search simulation system for medical literature using embedding and reranking models",
)

metadata_dataset = None
emb_id_to_metadata_id = {}  # Maps embedding index to metadata row index
embeddings_matrix = None  # Single concatenated matrix of all embeddings
# url_content_cache = {}  # Cache for URL content to improve /visit performance
sampling_params = None  # Will be initialized based on server mode


# FAISS-related variables
faiss_manager = None  # FaissIndexManager instance

# Cache instance
startup_cache = None  # StartupDataCache instance

# Server process references
embedding_server_process = None
reranker_server_process = None


def start_model_servers():
    """Start embedding and reranker servers using vllm serve"""
    global embedding_server_process, reranker_server_process
    
    logger.info("Starting VLLM model servers...")
    
    # Start embedding server using vllm serve
    embedding_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.EMBEDDING_MODEL_NAME,
        "--port", str(config.EMBEDDING_SERVER_PORT),
        "--host", config.EMBEDDING_SERVER_HOST,
        "--tensor-parallel-size", str(config.EMBEDDING_TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization", str(config.EMBEDDING_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.MAX_MODEL_LEN),
        "--trust-remote-code",
        "--served-model-name", "embedding-model",
        "--task", "embed",  # Specify embedding task
        "--hf-overrides", '{"is_matryoshka": true, "matryoshka_dimensions": [256, 512, 1024]}'
    ]
    logger.info(f"Starting VLLM embedding server: {' '.join(embedding_cmd)}")
    embedding_server_process = subprocess.Popen(
        embedding_cmd,
        stdout=subprocess.DEVNULL, # Discard stdout
        stderr=subprocess.DEVNULL,  # Discard stderr
        env={**os.environ, "CUDA_VISIBLE_DEVICES": config.EMBEDDING_GPU_DEVICES}
    )
    
    # Start reranker server using vllm serve
    # reranker_cmd = [
    #     sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    #     "--model", config.RERANKER_MODEL_NAME,
    #     "--port", str(config.RERANKER_SERVER_PORT),
    #     "--host", config.RERANKER_SERVER_HOST,
    #     "--tensor-parallel-size", str(config.RERANK_TENSOR_PARALLEL_SIZE),
    #     "--gpu-memory-utilization", str(config.RERANK_GPU_MEMORY_UTILIZATION),
    #     "--max-model-len", str(config.MAX_RERANK_LEN),
    #     "--trust-remote-code",
    #     "--served-model-name", "reranker-model",
    #     "--max_logprobs", str(config.RERANK_MAX_LOGPROBS),
    # ]
    # logger.info(f"Starting VLLM reranker server: {' '.join(reranker_cmd)}")
    # reranker_server_process = subprocess.Popen(
    #     reranker_cmd,
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL,
    #     env={**os.environ, "CUDA_VISIBLE_DEVICES": config.RERANK_GPU_DEVICES}
    # )
    
    # Wait for servers to be ready using VLLM's OpenAI-compatible health endpoint
    embedding_url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/health"
    reranker_url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/health"
    
    max_retries = config.MODEL_LOADING_TIMEOUT  # 600 seconds timeout
    for i in range(max_retries):
        time.sleep(1)
        try:
            # Check embedding server
            embed_resp = requests.get(embedding_url, timeout=1)
            embed_ready = embed_resp.status_code == 200
            # embed_ready = True
            
            # Check reranker server
            # rerank_resp = requests.get(reranker_url, timeout=1)
            # rerank_ready = rerank_resp.status_code == 200
            rerank_ready = True
            
            if embed_ready and rerank_ready:
                logger.info("Both VLLM servers are ready!")
                return True
                
        except requests.exceptions.RequestException:
            if i % 10 == 0:
                logger.info(f"Waiting for VLLM servers to start... ({i}s)")
            continue
    
    raise Exception("Failed to start VLLM servers within timeout")


def stop_model_servers():
    """Stop the model server subprocesses"""
    global embedding_server_process, reranker_server_process
    
    logger.info("Stopping model servers...")
    
    if embedding_server_process:
        embedding_server_process.terminate()
        embedding_server_process.wait(timeout=60)
        embedding_server_process = None
        
    if reranker_server_process:
        reranker_server_process.terminate()
        reranker_server_process.wait(timeout=60)
        reranker_server_process = None
        
    logger.info("Model servers stopped")


def get_max_passage_id(dataset):
    """
    Find the maximum passage_id in a dataset
    
    Args:
        dataset: Dataset object with 'passage_id' (list) column
        
    Returns:
        int: Maximum passage_id found
    """
    return max(dataset[-1]['passage_id'])


def offset_dataset_passage_ids(dataset, offset):
    """
    Offset all passage_ids in a dataset by a given amount
    
    Args:
        dataset: Dataset object with 'passage_id' (list) column
        offset: Integer offset to add to all passage_ids
        
    Returns:
        Dataset: New dataset with offset passage_ids
    """
    def offset_passage_ids(example):
        example["passage_id"] = [pid + offset for pid in example["passage_id"]]
        return example
    
    return dataset.map(offset_passage_ids, num_proc=16)


def concat_datasets_with_passage_id_offset(datasets, dataset_names=None):
    """
    Concatenate multiple datasets while ensuring passage_ids don't overlap
    
    Args:
        datasets: List of dataset objects
        dataset_names: Optional list of names for logging
        
    Returns:
        Dataset: Concatenated dataset with non-overlapping passage_ids
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    if len(datasets) == 1:
        return datasets[0]
    
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(len(datasets))]
    
    # Start with the first dataset
    combined_dataset = datasets[0]
    current_max_passage_id = get_max_passage_id(datasets[0])
    
    logger.info(f"{dataset_names[0]} has {len(datasets[0])} records, max passage_id: {current_max_passage_id}")
    
    # Process remaining datasets with offsets
    for i, dataset in enumerate(datasets[1:], 1):
        dataset_name = dataset_names[i] if i < len(dataset_names) else f"Dataset {i+1}"
        
        # Calculate offset for this dataset
        offset = current_max_passage_id + 1
        original_max = get_max_passage_id(dataset)
        
        logger.info(f"{dataset_name} has {len(dataset)} records, original max passage_id: {original_max}")
        logger.info(f"Offsetting {dataset_name} passage_ids by {offset}")
        
        # Apply offset and concatenate
        offset_dataset = offset_dataset_passage_ids(dataset, offset)
        combined_dataset = concatenate_datasets([combined_dataset, offset_dataset])
        
        # Update current max for next iteration
        current_max_passage_id = original_max + offset
        logger.info(f"After offset, {dataset_name} max passage_id: {current_max_passage_id}")
    
    logger.info(f"Combined dataset has {len(combined_dataset)} records")
    return combined_dataset


def get_embid2metadata_id(dataset):
    """
    Create a dictionary mapping passage_id to metadata row index from a dataset
    where passage_id is a list column.

    Args:
        dataset: Dataset object with 'passage_id' (list) column

    Returns:
        dict: Dictionary mapping passage_id -> metadata row index
    """
    passage_id_to_metadata_id = {}

    for idx, row in enumerate(dataset):
        passage_ids = row["passage_id"]  # This is a list

        # Map each passage_id in the list to the metadata row index
        for passage_id in passage_ids:
            passage_id_to_metadata_id[passage_id] = idx

    return passage_id_to_metadata_id


@app.on_event("startup")
async def startup_event():
    """Initialize model servers and load data on startup"""
    # global metadata_dataset, emb_id_to_metadata_id, embeddings_matrix, url_content_cache, startup_cache, faiss_manager
    global metadata_dataset, emb_id_to_metadata_id, embeddings_matrix, startup_cache, faiss_manager

    start_time = time.time()
    logger.info("Starting initialization...")
    if DEBUG_MODE:
        logger.info("DEBUG MODE ENABLED - Detailed logging active")

    try:
        # Initialize cache
        if config.USE_STARTUP_CACHE:
            startup_cache = StartupDataCache(config.CACHE_DIR)
            logger.info("Initialized startup data cache")
        else:
            logger.info("Startup caching disabled")
            startup_cache = None
        # Start the model servers
        logger.info("Starting model servers...")
        start_model_servers()
        
        # Register cleanup on exit
        atexit.register(stop_model_servers)
        
        def signal_handler(signum, frame):
            stop_model_servers()
            sys.exit(0)
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Try to load from cache first
        cache_loaded = False
        if startup_cache and not config.FORCE_CACHE_REBUILD:
            logger.info("Attempting to load data from cache...")
            cache_start_time = time.time()
            
            # cached_emb_mapping, cached_url_content = startup_cache.load_cache_data()
            cached_emb_mapping = startup_cache.load_cache_data()
            
            if cached_emb_mapping is not None:
                emb_id_to_metadata_id = cached_emb_mapping
                logger.info(f"Loaded embedding ID mapping from cache ({len(emb_id_to_metadata_id)} entries)")
                
                # # Load URL content cache if available
                # if cached_url_content is not None:
                #     url_content_cache = cached_url_content
                #     logger.info(f"Loaded URL content cache from cache ({len(url_content_cache)} entries)")
                if os.path.isfile(startup_cache.url_content_cache_db.index_path) and os.path.isfile(startup_cache.url_content_cache_db.data_path):
                    startup_cache.url_content_cache_db.load()
                    cache_loaded = True
                    logger.info(f"Successfully loaded all data from cache in {time.time() - cache_start_time:.2f} seconds")
            else:
                logger.info("Cache data not available or invalid, proceeding with normal loading")

        if not cache_loaded:
            logger.info("Loading data from source (cache not used)")

        # Load metadata datasets
        metadata_start_time = time.time()

        from datasets import Features, Sequence, Value
        target_features = Features({
            'paper_url': Value('string'),
            'paper_id': Value('string'), 
            'paper_title': Value('string'),
            'year': Value('float64'),
            'venue': Value('string'),
            'passage_text': Sequence(Value('string')),
            # 'specialty': Sequence(Value('string')),
            'passage_id': Sequence(Value('int64'))
        })

        metadata_datasets = []
        for mdn in config.METADATA_DATASET_NAME:
            md = load_dataset(mdn, split="train")
            try:
                md = md.remove_columns(['pid'])
            except Exception as e:
                logger.warning(f"Failed to remove pid from {mdn}") 

            try:
                md = md.remove_columns(['specialty'])
            except Exception as e:
                logger.warning(f"Failed to remove specialty from {mdn}")

            md = md.cast(target_features)
            metadata_datasets.append(md)

        metadata_dataset = concat_datasets_with_passage_id_offset(metadata_datasets, config.METADATA_DATASET_NAME)
        
        # Only create embedding ID mapping if not loaded from cache
        if not cache_loaded:
            logger.info(f"Creating embedding ID to metadata ID mappings..")
            emb_id_to_metadata_id = get_embid2metadata_id(metadata_dataset)
        
        logger.info(f"Loaded {len(metadata_dataset)} metadata records")
        logger.info(f"Total embedding ID to metadata ID mappings: {len(emb_id_to_metadata_id)}")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Metadata loaded in {time.time() - metadata_start_time:.2f} seconds")

        # Initialize FAISS manager
        faiss_manager = FaissIndexManager(embedding_dimension=config.EMBEDDING_DIMENSION)

        # Setup FAISS index using incremental loading
        # FAISS will load original embeddings in batches and handle its own quantization
        faiss_manager.setup_index_incremental(
            embedding_folder=config.EMBEDDING_FOLDER,
            max_files=config.MAX_EMBEDDING_FILES,
            index_type=config.FAISS_INDEX_TYPE,
            nlist=config.FAISS_NLIST,
            use_cosine=config.FAISS_USE_COSINE,
            gpu_devices=config.FAISS_GPU_DEVICES,
            save_path=config.FAISS_INDEX_PATH,
            load_path=config.FAISS_INDEX_PATH if os.path.exists(config.FAISS_INDEX_PATH) else None,
            hnsw_m=config.FAISS_HNSW_M,
            hnsw_efConstruction=config.FAISS_HNSW_EF_CONSTRUCTION
        )
        
        faiss_start_time = time.time()
        logger.info(f"FAISS setup completed in {time.time() - faiss_start_time:.2f} seconds")
        logger.info(f"FAISS stats: {faiss_manager.get_stats()}")
        
        # FAISS handles all embedding storage and memory management
        torch.cuda.empty_cache()
        logger.info("FAISS index setup completed and GPU memory cleared")
        
        # Preload URL content cache only if not loaded from cache
        if not cache_loaded:
        #     cache_start_time = time.time()
        #     logger.info("Preloading URL content cache using parallel processing...")
            
            # def process_item(item):
            #     """Process a single item to create URL content"""
            #     url = item.get("paper_url")
            #     if not url:
            #         return None, None
                
            #     # Get paper title
            #     paper_title = item.get("paper_title", "Untitled")
                
            #     # Concatenate all passage texts
            #     passage_texts = item.get("passage_text", [])
            #     if isinstance(passage_texts, list):
            #         full_content = "\n\n".join(str(text) for text in passage_texts if text)
            #     else:
            #         full_content = str(passage_texts) if passage_texts else ""
                
            #     # Combine title and content
            #     unified_content = f"# {paper_title}\n\n{full_content}".strip()
                
            #     return url, unified_content
            
            # # Process items in parallel using ThreadPoolExecutor
            # with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2)) as executor:
            #     # Submit all items for processing
            #     futures = []
            #     for idx, item in enumerate(metadata_dataset):
            #         futures.append(executor.submit(process_item, item))
                
            #     # Collect results as they complete
            #     cache_count = 0
            #     for future in futures:
            #         url, content = future.result()
            #         if url and content:
            #             url_content_cache[url] = content
            #             cache_count += 1
                        
            #             if cache_count % 10000 == 0:
            #                 logger.info(f"Cached {cache_count} URLs so far...")

            # logger.info(f"URL content cache preloaded with {cache_count} entries in {time.time() - cache_start_time:.2f} seconds")
            
            
            
            # if DEBUG_MODE:
            #     logger.info(f"DEBUG: Average cache entry size: {sum(len(content) for content in url_content_cache.values()) / len(url_content_cache):.0f} chars")
        
            # Save to cache if we loaded from source and caching is enabled
            if startup_cache:
                logger.info("Saving URL-Content to binary cache")
                startup_cache.url_content_cache_db.build_cache(metadata_dataset)

                logger.info("Saving data to cache for future startups...")
                save_start_time = time.time()
                try:
                    startup_cache.save_cache_data(
                        emb_id_to_metadata_id=emb_id_to_metadata_id,
                        # url_content_cache=url_content_cache
                    )
                    logger.info(f"Successfully saved all data to cache in {time.time() - save_start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to save data to cache: {e}")
                    # Don't fail startup if cache saving fails
        
        logger.info("Model initialization completed!")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total initialization time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down...")
    stop_model_servers()


def compute_similarity(
    query_embedding: torch.Tensor, document_embeddings: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between query and document embeddings using PyTorch"""
    # Normalize embeddings
    query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    doc_norms = document_embeddings / document_embeddings.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.matmul(doc_norms, query_norm.unsqueeze(-1)).squeeze(-1)
    return similarities


# Search functions removed - now using FAISS for all similarity search


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


async def get_query_embedding(query: str) -> torch.Tensor:
    """Get embedding for a query using VLLM's OpenAI-compatible embedding endpoint"""
    try:
        start_time = time.time()
        
        # Prepare request using OpenAI-compatible embeddings endpoint
        url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/v1/embeddings"
        
        # Format query with instruction for retrieval task
        task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        formatted_input = f"Instruct: {task_instruction}\nQuery: {query}"
        
        payload = {
            "model": "embedding-model",  # The served model name we specified
            "input": formatted_input,
            "encoding_format": "float",
            "dimensions": config.EMBEDDING_DIMENSION
        }
        
        # Send request to VLLM's OpenAI-compatible embeddings endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=60)
            response.raise_for_status()
        
        # Extract embedding from OpenAI-compatible response format
        result = response.json()
        embedding = torch.tensor(result["data"][0]["embedding"])
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Query embedding generated in {time.time() - start_time:.3f} seconds")
            logger.info(f"DEBUG: Query: '{query[:100]}...' -> Embedding shape: {embedding.shape}")
        else:
            logger.debug(f"Generated embedding for query: {query[:50]}...")
        return embedding
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        # Return a random embedding as fallback
        return torch.randn(config.EMBEDDING_DIMENSION)


async def get_text_embeddings_batch(texts: List[str]) -> List[torch.Tensor]:
    """Get embeddings for a batch of text chunks using VLLM's OpenAI-compatible embedding endpoint"""
    try:
        start_time = time.time()
        
        # Prepare request using OpenAI-compatible embeddings endpoint
        url = f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/v1/embeddings"
        
        payload = {
            "model": "embedding-model",
            "input": texts,  # Send list of texts for batch processing
            "encoding_format": "float",
            "dimensions": config.EMBEDDING_DIMENSION
        }
        
        # Send request to VLLM's OpenAI-compatible embeddings endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=60)
            response.raise_for_status()
        
        # Extract embeddings from OpenAI-compatible response format
        result = response.json()
        embeddings = [torch.tensor(item["embedding"]) for item in result["data"]]
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Batch embeddings generated in {time.time() - start_time:.3f} seconds for {len(texts)} texts")
        
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        # Return random embeddings as fallback
        return [torch.randn(config.EMBEDDING_DIMENSION) for _ in texts]


async def preprocess_text(text: str) -> List[str]:
    """
    Convert text to lowercase and extract words (removing punctuation).
    """
    # Convert to lowercase and keep only alphanumeric characters and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Split into words and remove empty strings
    return [word for word in text.split() if word]


async def calculate_similarity(query_words: set, sentence_words: set) -> float:
    """
    Calculate similarity score based on keyword overlap.
    """
    if not query_words or not sentence_words:
        return 0.0
    
    # Count matching keywords
    matches = len(query_words.intersection(sentence_words))
    
    # Calculate score (Jaccard similarity)
    total_unique = len(query_words.union(sentence_words))
    if total_unique == 0:
        return 0.0
    
    return matches / total_unique


# async def generate_preview(passage_text: str, paper_title: str, query_embedding: torch.Tensor, preview_char: int) -> str:
#     """Generate a preview of the most relevant chunk from a passage"""
#     if preview_char <= 0:
#         return ""
    
#     # If passage is shorter than preview_char, return the entire passage
#     if len(passage_text) <= preview_char:
#         return passage_text
    
#     preview_char = max(preview_char, config.MINIMUM_PREVIEW_CHAR)
#     try:
#         # Split passage into chunks of preview_char size (no overlap)
#         chunks = []
#         chunk_texts = []
        
#         for i in range(0, len(passage_text), preview_char):
#             chunk = passage_text[i:i + preview_char]
#             if len(chunk.strip()) > 10:  # Skip too short chunks
#                 chunks.append(chunk)
#                 # Format as title + chunk for embedding
#                 chunk_text = f"{paper_title}\n{chunk}"
#                 chunk_texts.append(chunk_text)
        
#         if not chunks:
#             return ""
        
#         # If only one chunk, return it
#         if len(chunks) == 1:
#             return chunks[0]
        
#         # Get embeddings for all chunks in batch
#         chunk_embeddings = await get_text_embeddings_batch(chunk_texts)
#         # Convert to tensor for similarity computation
#         chunk_embeddings_tensor = torch.stack(chunk_embeddings)
#         # Compute similarities with query embedding
#         similarities = compute_similarity(query_embedding, chunk_embeddings_tensor)
#         # Find the chunk with highest similarity
#         best_idx = torch.argmax(similarities).item()
        
#         return chunks[best_idx]
        
#     except Exception as e:
#         logger.error(f"Error generating preview: {e}")
#         # Fallback: return first preview_char characters
#         return passage_text[:preview_char]


async def generate_preview(passage_text: str, paper_title: str, query: str, preview_char: int) -> str:
    """Generate a preview of the most relevant chunk from a passage"""
    if preview_char <= 0:
        return ""
    
    # If passage is shorter than preview_char, return the entire passage
    if len(passage_text) <= preview_char:
        return passage_text
    
    preview_char = max(preview_char, config.MINIMUM_PREVIEW_CHAR)
    try:
        # Split passage into chunks of preview_char size (no overlap)
        chunks = []
        chunk_texts = []
        
        for i in range(0, len(passage_text), preview_char):
            chunk = passage_text[i:i + preview_char]
            if len(chunk.strip()) > 10:  # Skip too short chunks
                chunks.append(chunk)
                # Format as title + chunk for embedding
                chunk_text = f"{paper_title}\n{chunk}"
                chunk_texts.append(chunk_text)
        
        if not chunks:
            return ""
        
        # If only one chunk, return it
        if len(chunks) == 1:
            return chunks[0]
        
        query_words = set(await preprocess_text(query))
        processed_chunks = await asyncio.gather(*[preprocess_text(chunk) for chunk in chunk_texts])
        chunk_texts_words = [set(chunk) for chunk in processed_chunks]
        # Await all similarity calculations
        similarities = await asyncio.gather(*[
            calculate_similarity(query_words, chunk_words) 
            for chunk_words in chunk_texts_words
        ])
        # Find the chunk with highest similarity
        best_idx = np.argmax(similarities).item()
        
        return chunks[best_idx]
        
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        # Fallback: return first preview_char characters
        return passage_text[:preview_char]


def format_instruction(instruction, query, doc):
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    output = prefix + output.strip() + suffix
    return output


def get_passage_text(passage_id: int) -> str:
    """Get the text of a passage by its ID from the metadata dataset"""
    global metadata_dataset
    if not metadata_dataset:
        raise ValueError("Metadata dataset is not loaded")
    metadata_id = emb_id_to_metadata_id.get(passage_id)
    item = metadata_dataset[metadata_id]
    passage_text = ""
    for pid, text in zip(item.get("passage_id", []), item.get("passage_text", [])):
        if pid == passage_id:
            passage_text += text
            break
    passage_text = item.get("paper_title", "").strip() + "\n" + passage_text.strip()
    passage_text = passage_text[:config.RERANK_MAX_DOC_CHAR]  # Limit to max doc char
    return passage_text.strip()


def softmax(x, temp=1.0):
    """Apply softmax function with temperature scaling"""
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x = np.array(x) / temp
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


async def rerank_results(query: str, results: List[tuple], top_k: int = None) -> List[tuple]:
    """Rerank search results using the reranker server"""
    rerank_start_time = time.time()
    
    if top_k is None:
        top_k = config.TOP_K_RERANK
    
    if DEBUG_MODE:
        logger.info(f"DEBUG: Starting reranking for {len(results)} results, selecting top {top_k}")
    try:
        # Prepare input for reranker
        prep_start_time = time.time()
        rerank_texts = []
        for idx, (passage_id, score) in enumerate(results):
            # Create query-document pairs for reranking
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            doc_text = get_passage_text(passage_id)
            # Format the instruction text
            formatted_text = format_instruction(instruction, query, doc_text)
            rerank_texts.append(formatted_text)
            if DEBUG_MODE and idx == 0:
                logger.info(f"DEBUG: Sample reranker input length: {len(formatted_text)} chars")
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Prepared {len(rerank_texts)} inputs for reranking in {time.time() - prep_start_time:.3f} seconds")

        # Send request to VLLM's OpenAI-compatible completions endpoint for scoring
        url = f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/v1/completions"
        
        # Process each text for reranking using VLLM's scoring capability
        rerank_scores = []
        generate_start_time = time.time()
        
        # Batch process for efficiency
        batch_size = config.RERANK_BATCH_SIZE
        async with httpx.AsyncClient() as client:
            for i in range(0, len(rerank_texts), batch_size):
                batch_texts = rerank_texts[i:i+batch_size]
                
                # Use VLLM's completions endpoint with logprobs to get scores
                payload = {
                    "model": "reranker-model",
                    "prompt": batch_texts,
                    "max_tokens": 1,
                    "logprobs": config.RERANK_MAX_LOGPROBS,  # Request many logprobs to find yes/no tokens
                    "temperature": 0.6
                }
                
                response = await client.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                # Extract scores from logprobs for yes/no tokens
                for idx, choice in enumerate(result["choices"]):
                    if DEBUG_MODE and i + idx < 3:  # Log first 3 examples
                        logger.info(f"\nDEBUG: Reranking example {i + idx + 1}:")
                        logger.info(f"  Input prompt (first 200 chars): {batch_texts[idx][:200]}...")
                    
                    if choice.get("logprobs") and choice["logprobs"].get("top_logprobs"):
                        # Get the first token's top logprobs
                        if len(choice["logprobs"]["top_logprobs"]) > 0:
                            first_token_logprobs = choice["logprobs"]["top_logprobs"][0]
                            
                            # Log top 5 logprobs in debug mode
                            if DEBUG_MODE and i + idx < 3:
                                sorted_logprobs = sorted(first_token_logprobs.items(), key=lambda x: x[1], reverse=True)[:5]
                                logger.info(f"  Top 5 logprobs:")
                                for token, logprob in sorted_logprobs:
                                    logger.info(f"    Token: '{token}' -> Logprob: {logprob:.4f}")
                            
                            # Look for yes/no tokens - check various possible formats
                            yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES", "▁yes", "▁Yes"]
                            no_tokens = ["no", "No", "NO", " no", " No", " NO", "▁no", "▁No"]
                            
                            # Get token IDs if available (VLLM might provide token strings directly)
                            yes_logprob = -np.inf
                            no_logprob = -np.inf
                            
                            # Search through all available logprobs
                            for token, logprob in first_token_logprobs.items():
                                if token in yes_tokens:
                                    yes_logprob = max(yes_logprob, logprob)
                                elif token in no_tokens:
                                    no_logprob = max(no_logprob, logprob)
                            
                            # Apply softmax to get probabilities
                            logprobs = [yes_logprob, no_logprob]
                            probabilities = softmax(logprobs)
                            score = probabilities[0]  # Probability of "yes"
                            
                            # Log yes/no logprobs and final score in debug mode
                            if DEBUG_MODE and i + idx < 3:
                                logger.info(f"  Yes logprob: {yes_logprob:.4f}")
                                logger.info(f"  No logprob: {no_logprob:.4f}")
                                logger.info(f"  Final score (P(yes)): {score:.4f}")
                        else:
                            score = 0.5  # Default neutral score
                            if DEBUG_MODE and i + idx < 3:
                                logger.info(f"  No logprobs found for first token, using default score: {score}")
                    else:
                        score = 0.5  # Default neutral score if no logprobs
                        if DEBUG_MODE and i + idx < 3:
                            logger.info(f"  No logprobs in response, using default score: {score}")
                    rerank_scores.append(score)
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Reranker server responded in {time.time() - generate_start_time:.3f} seconds")
        
        # Combine results with scores
        new_results = []
        for (passage_id, score), rerank_score in zip(results, rerank_scores):
            new_results.append((passage_id, rerank_score))

        # Sort by score and return top_k
        reranked = sorted(new_results, key=lambda x: x[1] or 0, reverse=True)
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total reranking time: {time.time() - rerank_start_time:.3f} seconds")
            logger.info(f"DEBUG: Top 3 reranked scores: {[score for _, score in reranked[:3]]}")
        logger.debug(f"Reranked {len(results)} results, returning top {top_k}")
        return reranked[:top_k]

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return results[:top_k]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents based on a query.

    Args:
        query: The search query text
        use_reranker: Whether to apply reranking model (default: False)
        top_k: Number of search results to return (optional, defaults to MAX_SEARCH_RESULTS from config)
        preview_char: Number of preview characters to return for each result (-1 to skip preview generation)
                     Must be at least MINIMUM_PREVIEW_CHAR if specified

    Returns:
        A list of URLs with metadata, optionally reranked for relevance.
        Results are limited to min(top_k, MAX_SEARCH_RESULTS) when top_k is specified.
        Each result includes a preview of the most relevant chunk if preview_char is specified.
    """
    try:
        search_start_time = time.time()
        if not metadata_dataset:
            raise HTTPException(status_code=500, detail="Models not initialized")

        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Processing search query: {query}")
        if DEBUG_MODE:
            logger.info(f"DEBUG: Search request - use_reranker={request.use_reranker}")

        # Get query embedding
        embed_start = time.time()
        query_embedding = await get_query_embedding(query)
        if DEBUG_MODE:
            logger.info(f"DEBUG: Query embedding obtained in {time.time() - embed_start:.3f} seconds")

        # Search through embeddings to find similar documents
        logger.info("Searching through embeddings...")
        
        num_candidates = (
            min(request.top_k, config.MAX_SEARCH_RESULTS) * 4
            if request.use_reranker
            else int((min(request.top_k, config.MAX_SEARCH_RESULTS) * 1.2))
        )  # Get more candidates if using reranker to account for deduplication

        search_start = time.time()
        # Use FAISS multi-GPU search
        logger.info("Using FAISS multi-GPU search...")
        distances, indices = faiss_manager.search(
            query_embedding,
            k=min(num_candidates, config.FAISS_SEARCH_K),
            normalize_query=config.FAISS_USE_COSINE
        )
        # Convert FAISS results to the expected format: list of (index, score) tuples
        top_matches = [(indices[0][i], float(distances[0][i])) for i in range(len(indices[0]))]
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Embedding search completed in {time.time() - search_start:.3f} seconds, found {len(top_matches)} matches")

        # Apply reranking if requested
        if request.use_reranker:
            logger.info(f"Applying reranking to {len(top_matches)} passage results")
            rerank_start = time.time()
            top_matches = await rerank_results(query, top_matches)
            if DEBUG_MODE:
                logger.info(f"DEBUG: Reranking completed in {time.time() - rerank_start:.3f} seconds")
        else:
            logger.info(f"Skipping reranking, using embedding similarity scores")

        # Handle duplicated results based on paper_id - keep only highest scoring passage per paper
        logger.info(
            f"Deduplicating results by paper_id from {len(top_matches)} passages"
        )
        paper_best_matches = {}  # paper_id -> (passage_idx, score)

        for idx, score in top_matches:
            metadata_id = emb_id_to_metadata_id.get(idx)
            if metadata_id is not None:
                item = metadata_dataset[metadata_id]
                paper_id = item.get("paper_id")
                # Keep the highest scoring passage for each paper
                if (
                    paper_id not in paper_best_matches
                    or score > paper_best_matches[paper_id][1]
                ):
                    paper_best_matches[paper_id] = (idx, score)

        # Sort by score and take top results
        deduplicated_matches = sorted(
            paper_best_matches.values(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"After deduplication: {len(deduplicated_matches)} unique papers")

        # Get metadata for deduplicated matches
        search_results = []
        # Use custom top_k if provided, but cap at MAX_SEARCH_RESULTS
        max_results = min(request.top_k, config.MAX_SEARCH_RESULTS) if request.top_k is not None else config.MAX_SEARCH_RESULTS
        for idx, score in deduplicated_matches[: max_results]:
            metadata_id = emb_id_to_metadata_id.get(idx)
            if metadata_id is not None and metadata_id < len(metadata_dataset):
                item = metadata_dataset[metadata_id]
                
                # Generate preview if requested
                preview = ""
                if request.preview_char is not None and request.preview_char != -1:
                    # Get the passage text for this specific passage
                    passage_text = get_passage_text(idx)
                    paper_title = item.get("paper_title", "Unknown Title")
                    # preview = await generate_preview(passage_text, paper_title, query_embedding, request.preview_char)
                    preview = await generate_preview(passage_text, paper_title, query, request.preview_char)

                result = SearchResult(
                    url=item.get("paper_url", f"https://example.com/paper_{paper_id}"),
                    metadata={
                        "paper_id": item.get("paper_id", f"paper_{paper_id}"),
                        "paper_title": item.get("paper_title", "Unknown Title"),
                        "year": item.get("year", "Unknown Year"),
                        "venue": item.get("venue", "Unknown Venue"),
                        "specialty": item.get("specialty", []),
                    },
                    score=score,
                    preview=preview
                )
                search_results.append(result)

        # Clear GPU cache after search to free memory
        torch.cuda.empty_cache()

        logger.info(
            f"Returning {len(search_results)} search results for query: {query}"
        )
        
        if DEBUG_MODE:
            logger.info(f"DEBUG: Total search time: {time.time() - search_start_time:.3f} seconds")
        return SearchResponse(results=search_results)

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/visit", response_model=VisitResponse)
async def visit(request: VisitRequest):
    """
    Visit a URL and return its content with a realistic response structure.
    Now uses pre-cached content for improved performance.
    """
    try:
        if not startup_cache.url_content_cache_db:
            return VisitResponse(
                url=request.url, data="Error: Content cache not initialized", status_code=500
            )

        url = request.url.strip()
        if not url:
            return VisitResponse(
                url=request.url, data="Error: URL cannot be empty", status_code=400
            )

        logger.info(f"Processing visit request for URL: {url}")
        
        # Check cache first
        # if url in url_content_cache:
        #     unified_content = url_content_cache[url]
        #     logger.info(
        #         f"Successfully returning cached content for URL: {url} (length: {len(unified_content)} chars)"
        #     )
        unified_content = startup_cache.url_content_cache_db.get(url)
        if unified_content:
            logger.info(
                f"Successfully returning cached content for URL: {url} (length: {len(unified_content)} chars)"
            )
            return VisitResponse(url=url, data=unified_content, status_code=200)
        else:
            logger.info(f"URL not found in cache: {url}")
            return VisitResponse(url=url, data="404 Not Found", status_code=404)

    except Exception as e:
        logger.error(f"Error in visit endpoint: {e}")
        return VisitResponse(url=request.url, data=f"Error: {str(e)}", status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if model servers are running
    embedding_healthy = False
    reranker_healthy = False
    
    async with httpx.AsyncClient() as client:
        try:
            embed_resp = await client.get(
                f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}/health", 
                timeout=5
            )
            embedding_healthy = embed_resp.status_code == 200
        except:
            pass
        
        try:
            rerank_resp = await client.get(
                f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}/health", 
                timeout=5
            )
            reranker_healthy = rerank_resp.status_code == 200
        except:
            pass
    
    return {
        "status": "healthy" if embedding_healthy and reranker_healthy else "degraded",
        "servers": {
            "embedding_server": {
                "healthy": embedding_healthy,
                "url": f"http://{config.EMBEDDING_SERVER_HOST}:{config.EMBEDDING_SERVER_PORT}"
            },
            "reranker_server": {
                "healthy": reranker_healthy,
                "url": f"http://{config.RERANKER_SERVER_HOST}:{config.RERANKER_SERVER_PORT}"
            }
        },
        "data_loaded": {
            "metadata_dataset": metadata_dataset is not None,
            "embeddings_matrix": embeddings_matrix is not None,
            "embeddings_shape": (
                embeddings_matrix.shape if embeddings_matrix is not None else None
            ),
        },
        "cache_statistics": {
            # "url_content_cache_size": len(url_content_cache),
            # "total_cache_memory_mb": sum(len(content) for content in url_content_cache.values()) / (1024 * 1024) if url_content_cache else 0,
            # "cache_initialized": bool(url_content_cache),
            "startup_cache_enabled": config.USE_STARTUP_CACHE,
            "startup_cache_stats": startup_cache.get_cache_stats() if startup_cache else None,
        },
    }


@app.get("/cache")
async def cache_info():
    """Get cache information and statistics"""
    if not startup_cache:
        return {"error": "Startup caching is disabled"}
    
    return {
        "cache_enabled": config.USE_STARTUP_CACHE,
        "cache_dir": config.CACHE_DIR,
        "force_rebuild": config.FORCE_CACHE_REBUILD,
        "stats": startup_cache.get_cache_stats(),
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear startup cache"""
    if not startup_cache:
        return {"error": "Startup caching is disabled"}
    
    try:
        startup_cache.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear cache: {str(e)}"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Search Simulation API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search - Search for documents (optional: use_reranker)",
            "visit": "POST /visit - Get passages from a document URL",
            "health": "GET /health - Health check",
            "cache": "GET /cache - Get cache information",
            "cache/clear": "POST /cache/clear - Clear startup cache",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Medical Search Simulation API")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.API_HOST,
        help="Host to bind the API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.API_PORT,
        help="Port to bind the API server"
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        default=config.EMBEDDING_SERVER_PORT,
        help="Port for the embedding server (overrides config)"
    )
    parser.add_argument(
        "--reranking-port",
        type=int,
        default=config.RERANKER_SERVER_PORT,
        help="Port for the reranking server (overrides config)"
    )
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        DEBUG_MODE = True
        config.DEBUG_MODE = True
        # Update logging level to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        print("\n*** DEBUG MODE ENABLED - Detailed logging active ***\n")
    
    # Override config ports with command line arguments if provided
    if hasattr(args, 'embedding_port') and args.embedding_port != config.EMBEDDING_SERVER_PORT:
        config.EMBEDDING_SERVER_PORT = args.embedding_port
    if hasattr(args, 'reranking_port') and args.reranking_port != config.RERANKER_SERVER_PORT:
        config.RERANKER_SERVER_PORT = args.reranking_port
    
    uvicorn.run(app, host=args.host, port=args.port)
