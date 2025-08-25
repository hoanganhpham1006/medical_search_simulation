# Medical Search Simulation API

A high-performance medical literature search system that uses embedding-based search with reranking capabilities. The system processes medical research papers from PubMed, stores their embeddings, and provides a search API that finds relevant papers based on semantic similarity.

## Features

- **FAISS Multi-GPU Search**: Scalable similarity search across multiple GPUs using FAISS
- **Fast Semantic Search**: Utilizes Qwen3-Embedding-8B for generating high-quality document embeddings
- **Advanced Reranking**: Yes/no classification-based reranking with Qwen3-Reranker-8B for improved relevance
- **GPU Load Balancing**: Distributes FAISS index shards across all available GPUs to eliminate bottlenecks
- **Built-in FAISS Quantization**: IVFPQ index type provides automatic vector quantization (PQ32x8)
- **Multi-Server Architecture**: Separate embedding and reranking servers for better resource management
- **Index Persistence**: FAISS indexes are saved/loaded for fast startup times
- **Incremental Index Building**: Memory-efficient batch processing during index construction
- **Deduplication**: Returns one result per paper to avoid duplicates
- **RESTful API**: Simple and intuitive FastAPI interface with automatic documentation
- **FlashInfer Optimization**: Enhanced performance with FlashInfer-python for accelerated inference
- **Customizable Results**: Control number of results with top_k parameter
- **Smart Previews**: Generate relevant text previews showing the most pertinent passage chunks
- **Fully Async Architecture**: Non-blocking HTTP requests using httpx for optimal performance under load

## System Architecture

The system consists of three main components:
1. **Main API Server** (`api.py`) - FastAPI application serving on port 10000
2. **Embedding Server** (`embedding_server.py`) - VLLM server for Qwen3-Embedding-8B on port 10001
3. **Reranker Server** (`reranker_server.py`) - VLLM server for Qwen3-Reranker-8B on port 10002

### Data Flow
1. User sends search query to API
2. API gets query embedding from embedding server (async)
3. API performs FAISS multi-GPU similarity search on distributed index
4. Optionally reranks results using reranker server (async)
5. Returns deduplicated results (one result per paper)

### Async Architecture
- **Non-blocking HTTP**: All HTTP requests use httpx.AsyncClient for concurrent processing
- **Async Embedding**: Query and batch embedding generation don't block the event loop
- **Async Reranking**: Reranker requests processed asynchronously for better throughput
- **Async Health Checks**: Server health monitoring without blocking other operations
- **Concurrent Preview Generation**: Smart preview generation runs concurrently with other operations

### FAISS Multi-GPU Architecture
- **Index Distribution**: FAISS index automatically sharded across all available GPUs
- **IVFPQ Index**: Inverted File with Product Quantization for memory-efficient search
  - 32 subvectors with 8 bits each (PQ32x8) optimized for GPU shared memory
  - Provides built-in compression without separate quantization step
- **Cosine Similarity**: Normalized vectors with Inner Product for optimal similarity search
- **Index Persistence**: Pre-built indexes saved to disk for fast startup
- **Incremental Building**: Batch-wise index construction to handle large datasets
- **Auto-scaling**: Automatically utilizes all available GPUs (configurable)

### Memory Optimization
- **FAISS GPU Management**: Automatic GPU memory allocation and load balancing
- **Built-in Quantization**: IVFPQ index provides automatic vector compression
- **Incremental Loading**: Embeddings loaded and processed in batches to avoid OOM
- **Index Caching**: Pre-built FAISS indexes avoid rebuild time
- **Memory Efficiency**: Original embeddings cleared immediately after index construction

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd medical_search_simulation

# Install dependencies
pip install -r requirements.txt

# Run the API server
python api.py

# In another terminal, run the stress test
python stress_test.py --ccu 64 --duration 60
```

## Installation

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Building FAISS-GPU from Source

For optimal performance with multi-GPU support, it's recommended to build FAISS from source:

```bash
# Clone FAISS repository
git clone https://github.com/facebookresearch/faiss
cd faiss

# Install SWIG (required for Python bindings)
conda install -c conda-forge swig

# Install cmake and dependencies
conda install conda-forge::cmake
conda update cmake 
conda install mkl mkl-service
conda install -c conda-forge gflags


# Configure and build FAISS with GPU support
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89" .
make -C build -j1 faiss
make -C build -j1 swigfaiss

# Install Python bindings
cd build/faiss/python && python setup.py install
```

**Note**: 
- Adjust `CMAKE_CUDA_ARCHITECTURES` based on your GPU architecture (e.g., "70" for V100, "80" for A100, "89" for RTX 4090)
- The `-j1` flag builds with single thread to avoid memory issues; increase if you have sufficient RAM
- Ensure CUDA toolkit is installed and matches your GPU driver version

### Prepare Data

2. Prepare embeddings:
   - Configure embedding path in `EMBEDDING_FOLDER` setting
   - Configure number of embedding files in `MAX_EMBEDDING_FILES` setting

3. Configure environment (optional):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4  # Set to your available GPUs
export HF_HOME=/path/to/huggingface/cache
```

4. Run the API server:
```bash
python api.py
```

The main API server will start on `http://0.0.0.0:10000` by default.
The system will automatically start the embedding server (port 10001) and reranker server (port 10002).

## Testing

### Quick API Test

Once the server is running, you can test it with these curl commands:

1. **Check if the API is healthy:**
```bash
curl http://localhost:10000/health
```

2. **Search for medical documents:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "diabetes treatment", "use_reranker": true, "top_k": 10, "preview_char": 300}'
```

3. **Visit a specific document** (replace URL with one from search results):
```bash
curl -X POST "http://localhost:10000/visit" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/paper_123"}'
```

### Debug Mode

To run the server with detailed debug logging:
```bash
python api.py --debug
```

This will enable comprehensive logging including:
- FAISS index setup and multi-GPU distribution times
- FAISS incremental index building progress
- Model loading times
- Query embedding generation time
- FAISS similarity search performance
- Reranking preparation and scoring times
- Total API response time breakdowns

Additional command line options:
```bash
python api.py --debug --host 0.0.0.0 --port 10000
```

## API Endpoints

### Search for Documents

```bash
POST /search
```

**Request body:**
```json
{
    "query": "diabetes treatment guidelines",
    "use_reranker": true,
    "top_k": 10,
    "preview_char": 512
}
```

**Parameters:**
- `query` (required): Search query text
- `use_reranker` (optional): Whether to apply reranking (default: false)
- `top_k` (optional): Number of search results to return (default: MAX_SEARCH_RESULTS from config, max: 20)
- `preview_char` (optional): Number of preview characters to return for each result (default: -1 to skip, min: MINIMUM_PREVIEW_CHAR)

**Example curl command:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "diabetes treatment guidelines",
    "use_reranker": true,
    "top_k": 10,
    "preview_char": 512
  }'
```

**Example without reranking:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cardiovascular risk factors",
    "use_reranker": false,
    "top_k": 5
  }'
```

**Example with preview generation:**
```bash
curl -X POST "http://localhost:10000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning in healthcare",
    "preview_char": 300
  }'
```

**Response:**
```json
{
  "results": [
    {
      "url": "https://example.com/paper_123",
      "metadata": {
        "paper_id": "paper_123",
        "paper_title": "Recent Advances in Diabetes Management",
        "year": "2024",
        "venue": "Nature Medicine",
        "specialty": ["endocrinology", "internal medicine"]
      },
      "score": 0.95,
      "preview": "Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency. Recent advances in diabetes management have focused on personalized treatment approaches, continuous glucose monitoring systems, and novel pharmacological interventions including GLP-1 receptor agonists..."
    }
  ]
}
```

### Visit a Document

```bash
POST /visit
```

**Request body:**
```json
{
    "url": "https://example.com/paper_123"
}
```

**Example curl command:**
```bash
curl -X POST "http://localhost:10000/visit" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/paper_123"
  }'
```

**Response:**
```json
{
  "url": "https://example.com/paper_123",
  "data": "# Recent Advances in Diabetes Management\n\nDiabetes mellitus is a chronic metabolic disorder...",
  "status_code": 200
}
```

**Status codes:**
- 200: Success
- 404: Document not found
- 400: Invalid request
- 500: Server error

### Health Check

```bash
GET /health
```

**Example curl command:**
```bash
curl -X GET "http://localhost:10000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "embedding_model": true,
    "reranker_model": true,
    "metadata_dataset": true,
    "embeddings_matrix": true,
    "embeddings_shape": [2300000, 4096]
  }
}
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:10000/docs`
- ReDoc: `http://localhost:10000/redoc`

## Configuration

Edit `config.py` to customize:

### Model Configuration
```python
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
METADATA_DATASET_NAME = "hoanganhpham/Miriad_Pubmed_metadata"
```

### Server Configuration
```python
EMBEDDING_SERVER_PORT = 10001
RERANKER_SERVER_PORT = 10002
API_SERVER_PORT = 10000
EMBEDDING_GPU_DEVICES = "0"           # Single GPU for embedding
RERANK_GPU_DEVICES = "1,2,3,4,5,6,7"  # Multi-GPU for reranking
```

### Search Parameters
```python
MAX_SEARCH_RESULTS = 20      # Maximum results to return
TOP_K_RERANK = 10           # Results to rerank
EMBEDDING_DIMENSION = 4096   # Embedding vector dimension
FAISS_SEARCH_K = 1000       # Initial k for FAISS search before reranking
MINIMUM_PREVIEW_CHAR = 100   # Minimum preview character length
```

### FAISS Settings
```python
FAISS_INDEX_TYPE = "IVFPQ"  # Options: "Flat", "IVFFlat", "IVFPQ"
FAISS_NLIST = 1024          # Number of clusters for IVF indexes
FAISS_USE_COSINE = True     # Use cosine similarity
FAISS_GPU_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]  # GPU devices for FAISS
FAISS_INDEX_PATH = "/mnt/sharefs/tuenv/medical_search_cache/faiss_index.bin"
```

### File Paths
```python
EMBEDDING_FOLDER = "/path/to/embeddings/"
MAX_EMBEDDING_FILES = 785  # Adjust based on your dataset
```

### Debug Configuration
```python
DEBUG_MODE = False           # Set to True for detailed logging
LOG_LEVEL = "INFO"          # Default log level
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Performance Considerations

### Memory Requirements
- **RAM**: Varies based on dataset size and FAISS index type
- **GPU**: 16GB+ VRAM recommended for both embedding and reranking models
- **Storage**: Disk space depends on embedding dataset size

### Optimization Tips
1. **Use IVFPQ index**: Provides automatic vector compression with good search quality
2. **Pre-build index**: Save built index to disk to avoid rebuild on startup
3. **GPU allocation**: Distribute FAISS index and model servers across multiple GPUs
4. **Disable reranking**: Use `use_reranker=false` for faster but less accurate results
5. **Monitor startup time**: Initial index building takes 5-10 minutes but can be cached
6. **Debug mode**: Enable to identify performance bottlenecks
7. **Async architecture**: The fully async implementation provides better performance under concurrent load

## Testing

See the Testing section above for comprehensive testing options including unit tests and stress testing.

## Troubleshooting

### Stress Testing

The project includes a comprehensive stress testing tool that simulates multiple concurrent users:

```bash
# Run stress test with 64 concurrent users for 60 seconds
python stress_test.py --ccu 64 --duration 60

# Run with custom API URL
python stress_test.py --url http://192.168.0.11:10000 --ccu 64

# Skip pre-flight checks
python stress_test.py --skip-preflight --ccu 100 --duration 120
```

The stress test will:
1. Run pre-flight checks to ensure API health
2. Test all endpoints (health, search, visit)
3. Simulate realistic user behavior (70% search, 30% visit)
4. Report comprehensive metrics including:
   - Response times (min, max, mean, median, p95, p99)
   - Success/failure rates
   - Requests per second
   - Error breakdown

### Common Issues

1. **Out of GPU Memory**
   - Reduce number of GPUs in `FAISS_GPU_DEVICES`
   - Use IVFPQ index type for built-in compression
   - Ensure no other processes are using GPU

2. **Slow Startup**
   - Initial FAISS index building takes 5-10 minutes
   - Pre-built indexes are loaded from disk on subsequent runs
   - Consider using fewer embedding files for testing

3. **Model Server Startup Failures**
   - Check if ports 10001 and 10002 are available
   - Verify GPU assignments in `EMBEDDING_GPU_DEVICES` and `RERANK_GPU_DEVICES`
   - Ensure VLLM is properly installed

4. **Model Loading Errors**
   - Ensure Hugging Face cache is accessible
   - Check internet connection for model downloads
   - Verify CUDA is properly installed

## Development

### Code Structure
```
medical_search_simulation/
├── api.py                      # Main FastAPI application
├── faiss_index_manager.py      # FAISS multi-GPU index management
├── cache_utils.py              # Caching utilities for startup
├── config.py                   # Configuration settings
├── preprocess.py               # Data preprocessing utilities
├── vllm_generate_emb.py        # VLLM embedding generation
├── stress_test.py              # Comprehensive stress testing tool
├── requirements.txt            # Python dependencies
```

### Adding New Features
1. Modify data models in the Pydantic classes in `api.py`
2. Update endpoints and business logic
3. Add corresponding tests
4. Update configuration in `config.py` if needed
5. Consider impact on FAISS index and memory usage

## License

This project is for research and educational purposes.