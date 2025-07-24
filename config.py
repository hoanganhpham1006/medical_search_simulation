"""
Configuration file for Medical Search Simulation API
"""

import os

# Model Configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
METADATA_DATASET_NAME = "/mnt/sharefs/hoanganh/Wiki_Metadata_4096_chunks"
METADATA_DATASET_NAME_2 = "/mnt/sharefs/hoanganh/hfw_fineweb_edu_2526"
METADATA_DATASET_NAME_3 = "/mnt/sharefs/hoanganh/peS2o_full"
METADATA_DATASET_NAME_4 = "/mnt/sharefs/hoanganh/arxiv_ttabs"

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
MINIMUM_PREVIEW_CHAR = 256  # Minimum preview character length

# FAISS Configuration
FAISS_INDEX_TYPE = "IVFHNSW"  # Options: "Flat", "IVFFlat", "IVFPQ", "IVFHNSW" (IVFPQ provides built-in quantization)
FAISS_NLIST = 262144  # Number of clusters for IVF indexes
FAISS_USE_COSINE = True  # Use cosine similarity (normalized vectors with IP)
FAISS_GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]  # GPU devices for FAISS
FAISS_INDEX_PATH = "/mnt/sharefs/hoanganh/wiki_fineweb_search_cache/faiss_ivfhnsw_index.bin"  # Path to save/load FAISS index
FAISS_SEARCH_K = 1000  # Initial k for FAISS search before reranking

# HNSW Configuration (for IVFHNSW index type)
FAISS_HNSW_M = 32  # Number of bi-directional links for HNSW (default: 32, range: 4-128)
FAISS_HNSW_EF_CONSTRUCTION = 200  # Size of dynamic candidate list during construction (default: 200)

# Reranker Configuration
MAX_LOGPROBS = 8192  # Maximum number of log probabilities to return
RERANK_BATCH_SIZE = 32  # Batch size for reranking

# File Paths
EMBEDDING_FOLDER = "/mnt/sharefs/hoanganh/wiki_fineweb_emb/"
# MAX_EMBEDDING_FILES = 3129 # Wiki
# MAX_EMBEDDING_FILES = 9983 # Wiki +  Fineweb Edu
MAX_EMBEDDING_FILES = 15303

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_MODE = False  # Set to True to enable detailed debug logging

# Environment Variables (with defaults)
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
HF_HOME = os.getenv("HF_HOME", "/tmp/huggingface")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "/tmp/transformers")

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
RERANK_GPU_DEVICES = "4,5,6,7"  # GPU device(s) for reranker server

# Model server specific configurations
EMBEDDING_TENSOR_PARALLEL_SIZE = 8
EMBEDDING_GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 4096  # Maximum sequence length

RERANK_TENSOR_PARALLEL_SIZE = 4
RERANK_GPU_MEMORY_UTILIZATION = 0.5
MAX_RERANK_LEN = 32768  # Maximum sequence length for reranker
RERANK_MAX_LOGPROBS = 10000
RERANK_MAX_DOC_CHAR = 30000 # Roughly cut-off at 30k characters per document

# Cache Configuration
USE_STARTUP_CACHE = True  # Enable caching of startup data
CACHE_DIR = "/mnt/sharefs/hoanganh/wiki_fineweb_search_cache"  # Directory for cache files
FORCE_CACHE_REBUILD = False  # Force rebuilding cache even if valid
