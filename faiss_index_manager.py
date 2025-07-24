"""
FAISS Index Manager for Medical Search Simulation
Handles FAISS multi-GPU index creation, loading, and searching
"""
import faiss
import numpy as np
import torch
import logging
import os
import time
import pickle
from typing import List, Tuple, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor
import config

logger = logging.getLogger(__name__)

class FaissIndexManager:
    """
    Manages FAISS indexes for multi-GPU similarity search
    """
    
    def __init__(self, embedding_dimension: int = 4096):
        self.embedding_dimension = embedding_dimension
        self.cpu_index = None
        self.gpu_index = None
        self.index_lock = threading.Lock()
        self.num_gpus = faiss.get_num_gpus()
        logger.info(f"Initialized FAISS manager with {self.num_gpus} GPUs")
        
    def create_index(self, 
                     index_type: str = "IVFFlat", 
                     nlist: int = 1024,
                     use_cosine: bool = True,
                     hnsw_m: int = 32,
                     hnsw_efConstruction: int = 200) -> faiss.Index:
        """
        Create a FAISS index based on configuration
        
        Args:
            index_type: Type of index ("Flat", "IVFFlat", "IVFPQ", "IVFHNSW")
            nlist: Number of clusters for IVF indexes
            use_cosine: Whether to use cosine similarity (L2 otherwise)
            hnsw_m: Number of bi-directional links for HNSW (default: 32)
            hnsw_efConstruction: Size of dynamic candidate list for HNSW construction (default: 200)
            
        Returns:
            FAISS index instance
        """
        logger.info(f"Creating FAISS {index_type} index with dimension {self.embedding_dimension}")
        
        if use_cosine:
            # For cosine similarity, we'll normalize vectors and use IP (Inner Product)
            if index_type == "Flat":
                index = faiss.IndexFlatIP(self.embedding_dimension)
            elif index_type == "IVFFlat":
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "IVFPQ":
                quantizer = faiss.IndexFlatIP(self.embedding_dimension)
                # Use 32 subvectors with 8 bits each (PQ32x8) to fit GPU shared memory constraints
                m = 32  # number of subvectors (reduced from 64 to fit GPU memory)
                nbits = 8  # bits per subvector
                index = faiss.IndexIVFPQ(quantizer, self.embedding_dimension, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
            elif index_type == "IVFHNSW":
                # Create HNSW quantizer for fast cluster assignment
                quantizer = faiss.IndexHNSWFlat(self.embedding_dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                quantizer.hnsw.efConstruction = hnsw_efConstruction
                # Create IVF index with HNSW quantizer
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                logger.info(f"Created IVFHNSW index with M={hnsw_m}, efConstruction={hnsw_efConstruction}, nlist={nlist}")
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        else:
            # L2 distance
            if index_type == "Flat":
                index = faiss.IndexFlatL2(self.embedding_dimension)
            elif index_type == "IVFFlat":
                quantizer = faiss.IndexFlatL2(self.embedding_dimension)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, faiss.METRIC_L2)
            elif index_type == "IVFPQ":
                quantizer = faiss.IndexFlatL2(self.embedding_dimension)
                # Use 32 subvectors with 8 bits each (PQ32x8) to fit GPU shared memory constraints
                m = 32  # number of subvectors (reduced from 64 to fit GPU memory)
                nbits = 8  # bits per subvector
                index = faiss.IndexIVFPQ(quantizer, self.embedding_dimension, nlist, m, nbits, faiss.METRIC_L2)
            elif index_type == "IVFHNSW":
                # Create HNSW quantizer for fast cluster assignment
                quantizer = faiss.IndexHNSWFlat(self.embedding_dimension, hnsw_m, faiss.METRIC_L2)
                quantizer.hnsw.efConstruction = hnsw_efConstruction
                # Create IVF index with HNSW quantizer
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, faiss.METRIC_L2)
                logger.info(f"Created IVFHNSW index with M={hnsw_m}, efConstruction={hnsw_efConstruction}, nlist={nlist}")
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
        
        logger.info(f"Created {index_type} index: {index}")
        return index
    
    def create_index_factory(self, factory_string: str) -> faiss.Index:
        """
        Create a FAISS index using factory string notation
        
        Args:
            factory_string: FAISS factory string (e.g., "IVF1024(HNSW32),Flat", "IVF1024,PQ32", "HNSW32")
            
        Returns:
            FAISS index instance
        """
        logger.info(f"Creating FAISS index using factory string: {factory_string}")
        
        try:
            index = faiss.index_factory(self.embedding_dimension, factory_string)
            logger.info(f"Created factory index: {type(index).__name__}")
            return index
        except Exception as e:
            logger.error(f"Failed to create index with factory string '{factory_string}': {e}")
            raise
    
    def load_embeddings_from_files(self, 
                                   embedding_folder: str, 
                                   max_files: int = 3205,
                                   batch_size: int = 50,
                                   normalize: bool = True) -> np.ndarray:
        """
        Load embeddings from .pt files and return as numpy array
        
        Args:
            embedding_folder: Path to folder containing embedding files
            max_files: Maximum number of files to load
            batch_size: Number of files to process in parallel
            normalize: Whether to normalize embeddings for cosine similarity
            
        Returns:
            Normalized embeddings as numpy array
        """
        logger.info(f"Loading embeddings from {embedding_folder}")
        start_time = time.time()
        
        def load_embedding_file(file_idx: int) -> Tuple[int, Optional[torch.Tensor]]:
            """Load a single embedding file"""
            file_path = os.path.join(embedding_folder, f"embeddings_{file_idx}.pt")
            try:
                embeddings = torch.load(file_path, map_location="cpu")
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                return file_idx, embeddings
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                return file_idx, None
        
        # Load embeddings in parallel batches
        all_embeddings = []
        
        for batch_start in range(0, max_files, batch_size):
            batch_end = min(batch_start + batch_size, max_files)
            logger.info(f"Loading embedding batch {batch_start}-{batch_end-1}")
            
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
                futures = {executor.submit(load_embedding_file, i): i 
                          for i in range(batch_start, batch_end)}
                
                batch_embeddings = [None] * (batch_end - batch_start)
                for future in futures:
                    file_idx, embeddings = future.result()
                    if embeddings is not None:
                        batch_embeddings[file_idx - batch_start] = embeddings
                
                # Add non-None embeddings to collection
                for embeddings in batch_embeddings:
                    if isinstance(embeddings, torch.Tensor):
                        if embeddings.dim() == 2:
                            all_embeddings.extend(embeddings)
                        else:
                            all_embeddings.append(embeddings)
        
        if not all_embeddings:
            raise ValueError("No embeddings were loaded successfully")
        
        # Convert to numpy and stack
        logger.info("Converting embeddings to numpy array...")
        embeddings_matrix = torch.vstack(all_embeddings).numpy().astype(np.float32)
        
        # Normalize for cosine similarity if requested
        if normalize:
            logger.info("Normalizing embeddings for cosine similarity...")
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_matrix = embeddings_matrix / norms
        
        logger.info(f"Loaded embeddings matrix: {embeddings_matrix.shape} in {time.time() - start_time:.2f}s")
        return embeddings_matrix
    
    def build_and_train_index(self, 
                             embeddings: np.ndarray,
                             index_type: str = "IVFFlat",
                             nlist: int = 1024,
                             use_cosine: bool = True,
                             training_sample_size: Optional[int] = None,
                             hnsw_m: int = 32,
                             hnsw_efConstruction: int = 200) -> faiss.Index:
        """
        Build and train a FAISS index with the given embeddings
        
        Args:
            embeddings: Input embeddings array
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF indexes
            use_cosine: Whether to use cosine similarity
            training_sample_size: Number of vectors to use for training (None = use all)
            hnsw_m: Number of bi-directional links for HNSW (default: 32)
            hnsw_efConstruction: Size of dynamic candidate list for HNSW construction (default: 200)
            
        Returns:
            Trained FAISS index
        """
        logger.info(f"Building and training {index_type} index...")
        start_time = time.time()
        
        # Create the index
        index = self.create_index(index_type, nlist, use_cosine, hnsw_m, hnsw_efConstruction)
        
        # Train the index if needed (IVF-based indexes need training)
        if hasattr(index, 'train') and not index.is_trained:
            if training_sample_size and training_sample_size < len(embeddings):
                # Use a random sample for training to speed up the process
                logger.info(f"Using {training_sample_size} samples for training...")
                train_indices = np.random.choice(len(embeddings), training_sample_size, replace=False)
                train_data = embeddings[train_indices]
            else:
                train_data = embeddings
            
            logger.info(f"Training index with {len(train_data)} vectors...")
            index.train(train_data)
            logger.info("Index training completed")
        
        # Add all embeddings to the index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        logger.info(f"Index build completed in {time.time() - start_time:.2f}s")
        logger.info(f"Index contains {index.ntotal} vectors")
        
        return index
    
    def build_index_incremental(self,
                               embedding_folder: str,
                               max_files: int = 3205,
                               index_type: str = "IVFFlat",
                               nlist: int = 1024,
                               use_cosine: bool = True,
                               batch_size: int = 50,
                               training_samples_per_file: int = 512,
                               hnsw_m: int = 32,
                               hnsw_efConstruction: int = 200) -> faiss.Index:
        """
        Build FAISS index incrementally by loading embeddings in batches
        This avoids loading all embeddings into memory at once
        
        Args:
            embedding_folder: Path to folder containing embedding files
            max_files: Maximum number of files to load
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF indexes
            use_cosine: Whether to use cosine similarity
            batch_size: Number of files to process at once
            training_samples_per_file: Number of samples to collect per file for training
            hnsw_m: Number of bi-directional links for HNSW (default: 32)
            hnsw_efConstruction: Size of dynamic candidate list for HNSW construction (default: 200)
            
        Returns:
            Trained FAISS index
        """
        logger.info(f"Building {index_type} index incrementally from {max_files} files...")
        start_time = time.time()
        
        # Create the index
        index = self.create_index(index_type, nlist, use_cosine, hnsw_m, hnsw_efConstruction)
        
        # For IVF indexes, we need to train first
        if index_type in ["IVFFlat", "IVFPQ", "IVFHNSW"]:
            logger.info("Collecting training samples...")
            training_data = []
            
            # Collect training samples from a subset of files
            # Need ~10M training points for 262k clusters. With 512 samples per file, need ~20k files
            num_training_files = min(max_files, max_files)  # Use all available files for training
            step = max(1, max_files // 20000)  # Sample enough files to get 20k files total
            for i in range(0, num_training_files, step):
                file_path = os.path.join(embedding_folder, f"embeddings_{i}.pt")
                try:
                    embeddings = torch.load(file_path, map_location="cpu")
                    if embeddings.dim() == 1:
                        embeddings = embeddings.unsqueeze(0)
                    
                    # Convert to numpy
                    embeddings_np = embeddings.numpy().astype(np.float32)
                    
                    # Normalize if using cosine
                    if use_cosine:
                        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                        norms[norms == 0] = 1
                        embeddings_np = embeddings_np / norms
                    
                    # Sample some vectors for training
                    n_samples = min(training_samples_per_file, len(embeddings_np))
                    if len(embeddings_np) > n_samples:
                        indices = np.random.choice(len(embeddings_np), n_samples, replace=False)
                        training_data.append(embeddings_np[indices])
                    else:
                        training_data.append(embeddings_np)
                    
                except Exception as e:
                    logger.warning(f"Failed to load training samples from {file_path}: {e}")
            
            if training_data:
                training_data = np.vstack(training_data)
                logger.info(f"Training index with {len(training_data)} samples...")
                index.train(training_data)
                logger.info("Index training completed")
                # Clear training data to free memory
                del training_data
            else:
                logger.warning("No training data collected, index may not perform well")
        
        # Load and add embeddings in batches
        logger.info("Adding embeddings to index in batches...")
        total_added = 0
        
        def load_and_process_file(file_idx: int) -> Tuple[int, Optional[np.ndarray]]:
            """Load a single embedding file and convert to numpy"""
            file_path = os.path.join(embedding_folder, f"embeddings_{file_idx}.pt")
            try:
                embeddings = torch.load(file_path, map_location="cpu")
                if embeddings.dim() == 1:
                    embeddings = embeddings.unsqueeze(0)
                
                # Convert to numpy
                embeddings_np = embeddings.numpy().astype(np.float32)
                
                # Normalize if using cosine
                if use_cosine:
                    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    embeddings_np = embeddings_np / norms
                
                return file_idx, embeddings_np
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                return file_idx, None
        
        # Process files in batches
        for batch_start in range(0, max_files, batch_size):
            batch_end = min(batch_start + batch_size, max_files)
            logger.info(f"Processing embedding batch {batch_start}-{batch_end-1}")
            
            # Load files in parallel
            batch_embeddings = []
            with ThreadPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
                futures = [executor.submit(load_and_process_file, i) 
                          for i in range(batch_start, batch_end)]
                
                for future in futures:
                    file_idx, embeddings_np = future.result()
                    if embeddings_np is not None:
                        batch_embeddings.append(embeddings_np)
            
            # Add batch to index
            if batch_embeddings:
                batch_matrix = np.vstack(batch_embeddings)
                index.add(batch_matrix)
                total_added += len(batch_matrix)
                logger.info(f"Added {len(batch_matrix)} vectors (total: {total_added})")
                
                # Clear batch to free memory
                del batch_embeddings
                del batch_matrix
        
        logger.info(f"Incremental index build completed in {time.time() - start_time:.2f}s")
        logger.info(f"Index statistics: {index.ntotal} vectors")
        
        return index
    
    def _is_gpu_compatible(self, index: faiss.Index) -> bool:
        """
        Check if an index is compatible with GPU
        
        Args:
            index: FAISS index to check
            
        Returns:
            bool: True if GPU compatible, False otherwise
        """
        # Check if index contains HNSW
        index_str = str(type(index).__name__)
        if "HNSW" in index_str or hasattr(index, 'hnsw'):
            return False
            
        # Check if the index has an HNSW quantizer (for IVFHNSW)
        if hasattr(index, 'quantizer'):
            quantizer_str = str(type(index.quantizer).__name__)
            if "HNSW" in quantizer_str:
                return False
        
        return True
    
    def setup_multi_gpu_index(self, 
                             cpu_index: faiss.Index,
                             gpu_devices: Optional[List[int]] = None) -> faiss.Index:
        """
        Setup multi-GPU index from CPU index
        
        Args:
            cpu_index: Trained CPU index
            gpu_devices: List of GPU device IDs to use (None = use all available)
            
        Returns:
            Multi-GPU index (or CPU index if GPU not supported)
        """
        # Check if index contains HNSW
        index_str = str(type(cpu_index).__name__)
        logger.info(f"Index type: {index_str}")
        
        if "HNSW" in index_str or hasattr(cpu_index, 'hnsw'):
            logger.info("HNSW indexes are not supported on GPU, using CPU index")
            return cpu_index
            
        # Check if the index has an HNSW quantizer (for IVFHNSW)
        if hasattr(cpu_index, 'quantizer'):
            quantizer_str = str(type(cpu_index.quantizer).__name__)
            logger.info(f"Quantizer type: {quantizer_str}")
            if "HNSW" in quantizer_str:
                logger.info("Index with HNSW quantizer is not supported on GPU, using CPU index")
                return cpu_index
        
        if self.num_gpus == 0:
            logger.warning("No GPUs available, returning CPU index")
            return cpu_index
        
        if gpu_devices is None:
            # Use all available GPUs
            gpu_devices = list(range(self.num_gpus))
        
        logger.info(f"Setting up multi-GPU index on devices: {gpu_devices}")
        
        # Create multi-GPU index
        if len(gpu_devices) == 1:
            # Single GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_devices[0], cpu_index)
            logger.info(f"Created single GPU index on device {gpu_devices[0]}")
        else:
            # Multi-GPU using proper FAISS API
            # Create resources for each GPU
            resources = [faiss.StandardGpuResources() for _ in gpu_devices]
            
            # Create vectors for resources and devices (required by FAISS API)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            
            # Add resources and device IDs to vectors
            for gpu_id, res in zip(gpu_devices, resources):
                vdev.push_back(gpu_id)
                vres.push_back(res)
                logger.info(f"Added GPU resource for device {gpu_id}")
            
            # Configure cloner options for sharding
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # Split dataset across GPUs instead of replicating
            co.useFloat16 = True  # Use float16 to save memory
            co.usePrecomputed = False
            
            # For IVF indexes, we can share the quantizer
            if hasattr(cpu_index, 'quantizer'):
                co.common_ivf_quantizer = True
                logger.info("Enabled common IVF quantizer for multi-GPU index")
            
            # Convert to multi-GPU index
            gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu_index, co)
            
            # Keep references to prevent garbage collection
            gpu_index.referenced_objects = resources
            
            logger.info(f"Successfully created sharded multi-GPU index across devices: {gpu_devices}")
        
        logger.info(f"Multi-GPU index setup complete on {len(gpu_devices)} GPUs")
        return gpu_index
    
    def save_index(self, index: faiss.Index, filepath: str):
        """Save FAISS index to disk"""
        logger.info(f"Saving FAISS index to {filepath}")
        faiss.write_index(index, filepath)
        logger.info("Index saved successfully")
    
    def load_index(self, filepath: str) -> faiss.Index:
        """Load FAISS index from disk"""
        logger.info(f"Loading FAISS index from {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        index = faiss.read_index(filepath)
        logger.info(f"Loaded index with {index.ntotal} vectors")
        return index
    
    def search(self, 
               query_embeddings: Union[np.ndarray, torch.Tensor], 
               k: int = 100,
               normalize_query: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for similar vectors
        
        Args:
            query_embeddings: Query vectors (can be 1D or 2D)
            k: Number of nearest neighbors to return
            normalize_query: Whether to normalize query vectors
            
        Returns:
            Tuple of (distances, indices)
        """
        # Check if we have an index (either GPU or CPU)
        if self.gpu_index is None and self.cpu_index is None:
            raise ValueError("No index loaded. Call setup_index first.")
        
        # Use GPU index if available, otherwise use CPU index
        search_index = self.gpu_index if self.gpu_index is not None else self.cpu_index
        
        # Convert to numpy if needed
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()
        
        # Ensure 2D array
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize if requested
        if normalize_query:
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_embeddings = query_embeddings / norms
        
        # Perform search
        with self.index_lock:
            distances, indices = search_index.search(query_embeddings, k)
        
        return distances, indices
    
    def setup_index_incremental(self,
                               embedding_folder: str,
                               max_files: int,
                               index_type: str = "IVFFlat",
                               nlist: int = 1024,
                               use_cosine: bool = True,
                               gpu_devices: Optional[List[int]] = None,
                               save_path: Optional[str] = None,
                               load_path: Optional[str] = None,
                               hnsw_m: int = 32,
                               hnsw_efConstruction: int = 200) -> None:
        """
        Setup index incrementally from embedding files
        
        Args:
            embedding_folder: Path to folder containing embedding files
            max_files: Maximum number of files to load
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF
            use_cosine: Whether to use cosine similarity
            gpu_devices: GPU devices to use
            save_path: Path to save the built index
            load_path: Path to load existing index (skips building)
            hnsw_m: Number of bi-directional links for HNSW (default: 32)
            hnsw_efConstruction: Size of dynamic candidate list for HNSW construction (default: 200)
        """
        logger.info("Starting incremental FAISS index setup...")
        
        if load_path and os.path.exists(load_path):
            # Load existing index
            logger.info(f"Loading existing index from {load_path}")
            self.cpu_index = self.load_index(load_path)
        else:
            # Build new index incrementally
            self.cpu_index = self.build_index_incremental(
                embedding_folder=embedding_folder,
                max_files=max_files,
                index_type=index_type,
                nlist=nlist,
                use_cosine=use_cosine,
                hnsw_m=hnsw_m,
                hnsw_efConstruction=hnsw_efConstruction
            )
            
            # Save index if requested
            if save_path:
                self.save_index(self.cpu_index, save_path)
        
        # Setup multi-GPU index if supported
        if gpu_devices is not None and len(gpu_devices) > 0:
            # Also check the original index_type parameter for HNSW
            is_hnsw_config = "HNSW" in index_type
            if self._is_gpu_compatible(self.cpu_index) and not is_hnsw_config:
                self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
                logger.info("Multi-GPU index setup completed")
            else:
                if is_hnsw_config:
                    logger.info("HNSW index type from config - not compatible with GPU, using CPU index")
                else:
                    logger.info("Index type not compatible with GPU, using CPU index")
                self.gpu_index = self.cpu_index
    
    def setup_index_from_embeddings(self,
                                   embeddings: np.ndarray,
                                   index_type: str = "IVFFlat",
                                   nlist: int = 1024,
                                   use_cosine: bool = True,
                                   gpu_devices: Optional[List[int]] = None,
                                   save_path: Optional[str] = None,
                                   load_path: Optional[str] = None) -> None:
        """
        Setup index from pre-loaded embeddings array
        
        Args:
            embeddings: Pre-loaded embeddings array
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF
            use_cosine: Whether to use cosine similarity
            gpu_devices: GPU devices to use
            save_path: Path to save the built index
            load_path: Path to load existing index (skips building)
        """
        logger.info("Starting FAISS index setup from pre-loaded embeddings...")
        
        if load_path and os.path.exists(load_path):
            # Load existing index
            logger.info(f"Loading existing index from {load_path}")
            self.cpu_index = self.load_index(load_path)
        else:
            # Build new index from provided embeddings
            logger.info("Building new index from provided embeddings...")
            
            # Normalize embeddings if using cosine similarity
            if use_cosine:
                logger.info("Normalizing embeddings for cosine similarity...")
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings = embeddings / norms
            
            # Build and train index
            self.cpu_index = self.build_and_train_index(
                embeddings, 
                index_type=index_type,
                nlist=nlist,
                use_cosine=use_cosine,
                training_sample_size=min(100000, len(embeddings))  # Limit training data for speed
            )
            
            # Save index if requested
            if save_path:
                self.save_index(self.cpu_index, save_path)
        
        # Setup multi-GPU index if supported
        if gpu_devices is not None and len(gpu_devices) > 0 and self._is_gpu_compatible(self.cpu_index):
            self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
            logger.info("FAISS index setup completed successfully!")
            logger.info(f"Index ready with {self.gpu_index.ntotal} vectors on {self.num_gpus} GPUs")
        else:
            self.gpu_index = self.cpu_index
            logger.info("FAISS index setup completed successfully!")
            logger.info(f"Index ready with {self.cpu_index.ntotal} vectors (CPU mode)")

    def setup_index(self,
                   embedding_folder: str,
                   index_type: str = "IVFFlat",
                   nlist: int = 1024,
                   use_cosine: bool = True,
                   gpu_devices: Optional[List[int]] = None,
                   save_path: Optional[str] = None,
                   load_path: Optional[str] = None) -> None:
        """
        Complete index setup pipeline
        
        Args:
            embedding_folder: Path to embedding files
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF
            use_cosine: Whether to use cosine similarity
            gpu_devices: GPU devices to use
            save_path: Path to save the built index
            load_path: Path to load existing index (skips building)
        """
        logger.info("Starting FAISS index setup...")
        
        if load_path and os.path.exists(load_path):
            # Load existing index
            logger.info(f"Loading existing index from {load_path}")
            self.cpu_index = self.load_index(load_path)
        else:
            # Build new index
            logger.info("Building new index from embeddings...")
            
            # Load embeddings
            embeddings = self.load_embeddings_from_files(
                embedding_folder, 
                max_files=config.MAX_EMBEDDING_FILES,
                normalize=use_cosine
            )
            
            # Build and train index
            self.cpu_index = self.build_and_train_index(
                embeddings, 
                index_type=index_type,
                nlist=nlist,
                use_cosine=use_cosine,
                training_sample_size=min(100000, len(embeddings))  # Limit training data for speed
            )
            
            # Save index if requested
            if save_path:
                self.save_index(self.cpu_index, save_path)
        
        # Setup multi-GPU index if supported
        if gpu_devices is not None and len(gpu_devices) > 0 and self._is_gpu_compatible(self.cpu_index):
            self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
            logger.info("FAISS index setup completed successfully!")
            logger.info(f"Index ready with {self.gpu_index.ntotal} vectors on {self.num_gpus} GPUs")
        else:
            self.gpu_index = self.cpu_index
            logger.info("FAISS index setup completed successfully!")
            logger.info(f"Index ready with {self.cpu_index.ntotal} vectors (CPU mode)")
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        if self.gpu_index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "num_vectors": self.gpu_index.ntotal,
            "dimension": self.embedding_dimension,
            "num_gpus": self.num_gpus,
            "index_type": type(self.gpu_index).__name__
        }