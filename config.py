"""
Configuration file for Medical Search Simulation API
"""

import os

# Model Configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "/vast/llm/andy/search-r1/Wiki_Metadata_4096_chunks/data"
# File Paths
FAISS_INDEX_PATH = "/vast/llm/andy/search-r1/faiss_index.bin"  # Path to save/load FAISS index
EMBEDDING_FOLDER = "/vast/llm/andy/search-r1/Wiki_Metadata_4096_chunks_embs"
CACHE_DIR = "/vast/llm/andy/search-r1/cache"  # Directory for cache files
# HF_HOME = os.getenv("HF_HOME", "/vast/llm/share/")
# TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "/vast/llm/share/")

# VLLM Configuration
TRUST_REMOTE_CODE = True

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 10000
API_TITLE = "Wiki Search Simulation API"
API_VERSION = "1.0.0"

# Search Configuration
MAX_SEARCH_RESULTS = 20
TOP_K_RERANK = 10
EMBEDDING_DIMENSION = 2560  # Adjust based on actual model dimension
BATCH_SIZE = 65536  # Batch size for GPU processing during search
USE_GPU_PRELOADING = True  # Enable GPU preloading for better pipeline efficiency
PRELOAD_BUFFER_SIZE = 2  # Number of batches to keep in GPU buffer
MINIMUM_PREVIEW_CHAR = 256  # Minimum preview character length

# FAISS Configuration
FAISS_INDEX_TYPE = "IVFPQ"  # Options: "Flat", "IVFFlat", "IVFPQ" (IVFPQ provides built-in quantization)
FAISS_NLIST = 1024  # Number of clusters for IVF indexes
FAISS_USE_COSINE = True  # Use cosine similarity (normalized vectors with IP)
FAISS_GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]  # GPU devices for FAISS
FAISS_SEARCH_K = 1000  # Initial k for FAISS search before reranking

# Note: Quantization is now handled by FAISS internally (IVFPQ index type)

# Reranker Configuration
MAX_LOGPROBS = 8192  # Maximum number of log probabilities to return
RERANK_BATCH_SIZE = 32  # Batch size for reranking

# MAX_EMBEDDING_FILES = 3205  # Define correctly number of emb files
# 0 -> 281: Miriad, 282 -> 3204: Pubmed, 3205 ->: Taxonomy
MAX_EMBEDDING_FILES = 3129
# 0 -> 785: Wiki EN
# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_MODE = False  # Set to True to enable detailed debug logging

# Environment Variables (with defaults)
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# Model Loading Timeout (seconds)
MODEL_LOADING_TIMEOUT = 600  # 10 minutes

# API Timeouts
REQUEST_TIMEOUT = 30  # seconds
SEARCH_TIMEOUT = 60  # seconds
VISIT_TIMEOUT = 30  # seconds

# Server Configuration for separate model servers
EMBEDDING_SERVER_HOST = "127.0.0.1"
EMBEDDING_SERVER_PORT = 10001
RERANKER_SERVER_HOST = "127.0.0.1"
RERANKER_SERVER_PORT = 10002

# GPU allocation for separate servers
EMBEDDING_GPU_DEVICES = "0,1,2,3,4,5,6,7"  # GPU device(s) for embedding server
RERANK_GPU_DEVICES = "1,2,3,4,5,6,7"  # GPU device(s) for reranker server

# Model server specific configurations
EMBEDDING_TENSOR_PARALLEL_SIZE = 8
EMBEDDING_DATA_PARALLEL_SIZE = 1
EMBEDDING_GPU_MEMORY_UTILIZATION = 0.1
MAX_MODEL_LEN = 4096  # Maximum sequence length

RERANK_TENSOR_PARALLEL_SIZE = 1
RERANK_DATA_PARALLEL_SIZE = 7
RERANK_GPU_MEMORY_UTILIZATION = 0.4
MAX_RERANK_LEN = 32768  # Maximum sequence length for reranker
RERANK_MAX_LOGPROBS = 10000
RERANK_MAX_DOC_CHAR = 30000 # Roughly cut-off at 30k characters per document

# Cache Configuration
USE_STARTUP_CACHE = True  # Enable caching of startup data
FORCE_CACHE_REBUILD = False  # Force rebuilding cache even if valid
