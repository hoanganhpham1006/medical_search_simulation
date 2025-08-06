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


def load_embedding_file(file_path: str, truncate_dim: Optional[int] = None) -> Optional[torch.Tensor]:
    """Load a single embedding file"""
    try:
        embeddings = torch.load(file_path, map_location="cpu")
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if truncate_dim is not None:
            embeddings = embeddings[:, :truncate_dim]
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


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
                     use_cosine: bool = True) -> faiss.Index:
        """
        Create a FAISS index based on configuration
        
        Args:
            index_type: Type of index ("Flat", "IVFFlat", "IVFPQ")
            nlist: Number of clusters for IVF indexes
            use_cosine: Whether to use cosine similarity (L2 otherwise)
            
        Returns:
            FAISS index instance
        """
        assert index_type in ["Flat", "IVFFlat", "IVFPQ"], f"Unsupported index type: {index_type}"
        metric = faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2
        logger.info(f"Creating FAISS {index_type} index with dimension {self.embedding_dimension}")

        if index_type == "Flat":
            index = faiss.IndexFlatIP(self.embedding_dimension)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist, metric)
        elif index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            # Use 32 subvectors with 8 bits each (PQ32x8) to fit GPU shared memory constraints
            m = 32  # number of subvectors (reduced from 64 to fit GPU memory)
            nbits = 8  # bits per subvector
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dimension, nlist, m, nbits, metric)

        logger.info(f"Created {index_type} index: {index}")
        return index
    
    def load_embeddings_from_files(self, 
                                   files: List[str],
                                   normalize: bool = True) -> np.ndarray:
        """
        Load embeddings from .pt files and return as numpy array
        
        Args:
            files: List of embedding file paths
            normalize: Whether to normalize embeddings for cosine similarity
            
        Returns:
            Normalized embeddings as numpy array
        """
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=min(len(files), 32, os.cpu_count() * 2)) as executor:
            futures = [executor.submit(load_embedding_file, file_path, self.embedding_dimension) for file_path in files]
            all_embeddings = [future.result() for future in futures]
            all_embeddings = [embeddings for embeddings in all_embeddings if embeddings is not None]
        
        if not all_embeddings:
            raise ValueError("No embeddings were loaded successfully")
        
        embeddings_matrix = torch.vstack(all_embeddings).to(dtype=torch.float32)
        
        if normalize:
            embeddings_matrix = torch.nn.functional.normalize(embeddings_matrix, dim=1)
        
        embeddings_matrix = embeddings_matrix.numpy()
        logger.info(f"Loaded embeddings matrix: {embeddings_matrix.shape} in {time.time() - start_time:.2f}s")
        return embeddings_matrix
    
    def build_and_train_index(self, 
                             embeddings: np.ndarray,
                             index_type: str = "IVFFlat",
                             nlist: int = 1024,
                             use_cosine: bool = True,
                             training_sample_size: Optional[int] = None) -> faiss.Index:
        """
        Build and train a FAISS index with the given embeddings
        
        Args:
            embeddings: Input embeddings array
            index_type: Type of FAISS index
            nlist: Number of clusters for IVF indexes
            use_cosine: Whether to use cosine similarity
            training_sample_size: Number of vectors to use for training (None = use all)
            
        Returns:
            Trained FAISS index
        """
        logger.info(f"Building and training {index_type} index...")
        start_time = time.time()
        
        # Create the index
        index = self.create_index(index_type, nlist, use_cosine)
        
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
                               training_samples_per_file: int = 256) -> faiss.Index:
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
            
        Returns:
            Trained FAISS index
        """
        logger.info(f"Building {index_type} index incrementally from {max_files} files...")
        start_time = time.time()
        
        # Create the index
        index = self.create_index(index_type, nlist, use_cosine)
        
        # For IVF indexes, we need to train first
        if index_type in ["IVFFlat", "IVFPQ"]:
            logger.info("Collecting training samples...")
            
            # Collect training files (every 3rd file from first 1000)
            num_training_files = min(1000, max_files)
            training_files = []
            for i in range(0, num_training_files, 3):
                file_path = os.path.join(embedding_folder, f"embeddings_{i}.pt")
                if os.path.exists(file_path):
                    training_files.append(file_path)
            
            if training_files:
                try:
                    # Load all training embeddings using existing method
                    training_data = self.load_embeddings_from_files(
                        files=training_files, 
                        normalize=use_cosine
                    )
                    
                    # Sample training vectors if we have too many
                    max_training_samples = training_samples_per_file * len(training_files)
                    if len(training_data) > max_training_samples:
                        indices = np.random.choice(len(training_data), max_training_samples, replace=False)
                        training_data = training_data[indices]
                    
                    logger.info(f"Training index with {len(training_data)} samples...")
                    index.train(training_data)
                    logger.info("Index training completed")
                    
                    # Clear training data to free memory
                    del training_data
                    
                except Exception as e:
                    logger.warning(f"Failed to load training data: {e}")
                    logger.warning("Index may not perform well without training")
            else:
                logger.warning("No training data collected, index may not perform well")
        
        # Load and add embeddings in batches
        logger.info("Adding embeddings to index in batches...")
        total_added = 0
                
        # Process files in batches
        for batch_start in range(0, max_files, batch_size):
            batch_end = min(batch_start + batch_size, max_files)
            logger.info(f"Processing embedding batch {batch_start}-{batch_end-1}")
            
            batch_files = [os.path.join(embedding_folder, f"embeddings_{i}.pt") for i in range(batch_start, batch_end)]
            batch_embeddings = self.load_embeddings_from_files(batch_files)
            
            # Add batch to index
            index.add(batch_embeddings)
            total_added += len(batch_embeddings)
            logger.info(f"Added {len(batch_embeddings)} vectors (total: {total_added})")
                
            # Clear batch to free memory
            del batch_embeddings
        
        logger.info(f"Incremental index build completed in {time.time() - start_time:.2f}s")
        logger.info(f"Index statistics: {index.ntotal} vectors")
        
        return index
    
    def setup_multi_gpu_index(self, 
                             cpu_index: faiss.Index,
                             gpu_devices: Optional[List[int]] = None) -> faiss.Index:
        """
        Setup multi-GPU index from CPU index
        
        Args:
            cpu_index: Trained CPU index
            gpu_devices: List of GPU device IDs to use (None = use all available)
            
        Returns:
            Multi-GPU index
        """
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
        if self.gpu_index is None:
            raise ValueError("No index loaded. Call setup_index first.")
        
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
            distances, indices = self.gpu_index.search(query_embeddings, k)
        
        return distances, indices
    
    def setup_index_incremental(self,
                               embedding_folder: str,
                               max_files: int,
                               index_type: str = "IVFFlat",
                               nlist: int = 1024,
                               use_cosine: bool = True,
                               gpu_devices: Optional[List[int]] = None,
                               save_path: Optional[str] = None,
                               load_path: Optional[str] = None) -> None:
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
                use_cosine=use_cosine
            )
            
            # Save index if requested
            if save_path:
                self.save_index(self.cpu_index, save_path)
        
        # Setup multi-GPU index
        if gpu_devices is not None and len(gpu_devices) > 0:
            self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
            logger.info("Multi-GPU index setup completed")
    
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
        
        # Setup multi-GPU index
        self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
        
        logger.info("FAISS index setup completed successfully!")
        logger.info(f"Index ready with {self.gpu_index.ntotal} vectors on {self.num_gpus} GPUs")

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
        
        # Setup multi-GPU index
        self.gpu_index = self.setup_multi_gpu_index(self.cpu_index, gpu_devices)
        
        logger.info("FAISS index setup completed successfully!")
        logger.info(f"Index ready with {self.gpu_index.ntotal} vectors on {self.num_gpus} GPUs")
    
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