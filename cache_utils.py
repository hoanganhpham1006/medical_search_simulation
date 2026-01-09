"""
Cache utilities for Medical Search Simulation API
Handles caching of startup data to reduce loading time
"""

import os
import pickle
import torch
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import config
from contextlib import contextmanager
import struct
import mmap

logger = logging.getLogger(__name__)


class UltraFastBinaryCache:
    def __init__(self, data_path="cache.bin", index_path="cache.idx"):
        self.data_path = Path(data_path)
        self.index_path = Path(index_path)
        self.index: Dict[str, Tuple[int, int]] = {}
    
    def build_cache(self, metadata_dataset, show_progress=True):
        """Extremely fast sequential write"""
        
        logger.info("Building ultra-fast binary cache...")
        
        with open(self.data_path, 'wb') as data_file:
            offset = 0
            
            for i, item in enumerate(metadata_dataset):
                url = item.get("paper_url")
                if not url:
                    continue
                
                # Process content
                paper_title = item.get("paper_title", "Untitled")
                passage_texts = item.get("passage_text", [])
                
                if isinstance(passage_texts, list):
                    full_content = "\n\n".join(str(text) for text in passage_texts if text)
                else:
                    full_content = str(passage_texts) if passage_texts else ""
                
                unified_content = f"# {paper_title}\n\n{full_content}".strip()
                content_bytes = unified_content.encode('utf-8')
                
                # Write to file
                length = len(content_bytes)
                data_file.write(struct.pack('I', length))
                data_file.write(content_bytes)
                
                # Update index
                self.index[url] = (offset, length)
                offset += 4 + length  # 4 bytes for length + content
                
                if show_progress and (i + 1) % 10000000 == 0:
                    logger.info(f"Processed {i+1:,} URLs...")
        
        # Save index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Cache complete! {len(self.index):,} URLs indexed")
    
    def load(self):
        """Load for reading"""
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        
        self.data_file = open(self.data_path, 'rb')
        self.mmap = mmap.mmap(self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def get(self, url):
        if url not in self.index:
            return None
        
        offset, length = self.index[url]
        self.mmap.seek(offset + 4)  # Skip length header
        return self.mmap.read(length).decode('utf-8')


class StartupDataCache:
    """Manages caching for startup data including embeddings, metadata mappings, and URL content"""
    
    def __init__(self, cache_dir: str = "/tmp/medical_search_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.emb_id_mapping_file = self.cache_dir / "emb_id_to_metadata_id.pkl"
        # self.url_content_cache_file = self.cache_dir / "url_content_cache.pkl"
        self.url_content_cache_db = UltraFastBinaryCache(data_path = self.cache_dir / "url_content_cache.bin", index_path = self.cache_dir / "url_content_cache.idx")
        self.cache_info_file = self.cache_dir / "cache_info_new.json"
        
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_info(self) -> Dict[str, Any]:
        """Get current cache information and configuration hash"""
        try:
            if self.cache_info_file.exists():
                with open(self.cache_info_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read cache info: {e}")
        return {}
    
    def _save_cache_info(self, info: Dict[str, Any]):
        """Save cache information"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache info: {e}")
    
    def _get_config_hash(self) -> str:
        """Generate hash of relevant configuration parameters"""
        config_params = {
            'EMBEDDING_FOLDER': config.EMBEDDING_FOLDER,
            'MAX_EMBEDDING_FILES': config.MAX_EMBEDDING_FILES,
            'METADATA_DATASET_NAME': config.METADATA_DATASET_NAME,
            'EMBEDDING_DIMENSION': config.EMBEDDING_DIMENSION,
            'FAISS_INDEX_TYPE': config.FAISS_INDEX_TYPE,
            'FAISS_NLIST': config.FAISS_NLIST,
            'DEBUG_MODE': True
        }
        config_str = json.dumps(config_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def is_cache_valid(self) -> bool:
        """Check if cached data is valid based on configuration and file timestamps"""
        try:
            cache_info = self._get_cache_info()
            current_config_hash = self._get_config_hash()

            
            # Check if config has changed
            if cache_info.get('config_hash') != current_config_hash:
                logger.info("Cache invalid: Configuration has changed")
                return False
            
            # Check if all required cache files exist
            required_files = [
                self.emb_id_mapping_file,
                # self.url_content_cache_file
                self.url_content_cache_db.data_path,
                self.url_content_cache_db.index_path,
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    logger.info(f"Cache invalid: Missing file {file_path}")
                    return False
            
            # Check if embedding source files are newer than cache
            cache_timestamp = cache_info.get('timestamp', 0)
            embedding_folder = Path(config.EMBEDDING_FOLDER)
            
            if embedding_folder.exists():
                # Check a few sample embedding files
                for i in [0, 100, 500, min(1000, config.MAX_EMBEDDING_FILES - 1)]:
                    emb_file = embedding_folder / f"embeddings_{i}.pt"
                    if emb_file.exists() and emb_file.stat().st_mtime > cache_timestamp:
                        logger.info(f"Cache invalid: Embedding file {emb_file} is newer than cache")
                        return False
            
            logger.info("Cache is valid")
            return True
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def save_emb_id_mapping(self, emb_id_to_metadata_id: Dict[int, int]):
        """Save embedding ID to metadata ID mapping"""
        try:
            logger.info(f"Saving embedding ID mapping ({len(emb_id_to_metadata_id)} entries)")
            with open(self.emb_id_mapping_file, 'wb') as f:
                pickle.dump(emb_id_to_metadata_id, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved embedding ID mapping to {self.emb_id_mapping_file}")
        except Exception as e:
            logger.error(f"Failed to save embedding ID mapping: {e}")
    
    def load_emb_id_mapping(self) -> Optional[Dict[int, int]]:
        """Load embedding ID to metadata ID mapping"""
        try:
            if self.emb_id_mapping_file.exists():
                logger.info("Loading embedding ID mapping from cache")
                with open(self.emb_id_mapping_file, 'rb') as f:
                    mapping = pickle.load(f)
                logger.info(f"Loaded embedding ID mapping ({len(mapping)} entries)")
                return mapping
        except Exception as e:
            logger.error(f"Failed to load embedding ID mapping: {e}")
        return None
    
    # Quantization cache methods removed - FAISS handles quantization internally
    
    # def save_url_content_cache(self, url_content_cache: Dict[str, str]):
    #     """Save URL content cache"""
    #     try:
    #         logger.info(f"Saving URL content cache ({len(url_content_cache)} entries)")
    #         with open(self.url_content_cache_file, 'wb') as f:
    #             pickle.dump(url_content_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    #         # Calculate total size
    #         total_size = sum(len(content) for content in url_content_cache.values())
    #         logger.info(f"Saved URL content cache to {self.url_content_cache_file} ({total_size / (1024*1024):.1f} MB)")
            
    #     except Exception as e:
    #         logger.error(f"Failed to save URL content cache: {e}")
    
    # def load_url_content_cache(self) -> Optional[Dict[str, str]]:
    #     """Load URL content cache"""
    #     try:
    #         if self.url_content_cache_file.exists():
    #             logger.info("Loading URL content cache from cache")
    #             with open(self.url_content_cache_file, 'rb') as f:
    #                 cache = pickle.load(f)
                
    #             total_size = sum(len(content) for content in cache.values())
    #             logger.info(f"Loaded URL content cache ({len(cache)} entries, {total_size / (1024*1024):.1f} MB)")
    #             return cache
                
    #     except Exception as e:
    #         logger.error(f"Failed to load URL content cache: {e}")
    #     return None
    
    def save_cache_data(self, emb_id_to_metadata_id: Dict[int, int], 
                       url_content_cache: Optional[Dict[str, str]] = None):
        """Save cache data and update cache info"""
        try:
            # Save individual components
            self.save_emb_id_mapping(emb_id_to_metadata_id)
            
            # if url_content_cache is not None:
            #     self.save_url_content_cache(url_content_cache)
            
            # Update cache info
            cache_info = {
                'timestamp': int(os.path.getmtime(self.emb_id_mapping_file)),
                'config_hash': self._get_config_hash(),
                'debug_mode': config.DEBUG_MODE
            }
            self._save_cache_info(cache_info)
            
            logger.info("Successfully saved all cache data")
            
        except Exception as e:
            logger.error(f"Failed to save cache data: {e}")
    
    def load_cache_data(self) -> Tuple[Optional[Dict[int, int]], Optional[Dict[str, str]]]:
        """Load cached data"""
        if not self.is_cache_valid():
            logger.info("Cache is invalid, will not load")
            return None
        
        try:
            logger.info("Loading data from cache")
            
            # Load mapping
            emb_id_mapping = self.load_emb_id_mapping()
            
            # # Load URL content cache
            # url_content_cache = self.load_url_content_cache()
            
            # return emb_id_mapping, url_content_cache
            return emb_id_mapping
            
        except Exception as e:
            logger.error(f"Failed to load cache data: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached files"""
        try:
            logger.info("Clearing cache")
            for file_path in [
                self.emb_id_mapping_file,
                self.url_content_cache_db.data_path,
                self.url_content_cache_db.index_path,
                self.cache_info_file
            ]:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed {file_path}")
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'cache_dir': str(self.cache_dir),
            'cache_valid': self.is_cache_valid(),
            'files': {}
        }

        for name, file_path in [
            ('emb_id_mapping', self.emb_id_mapping_file),
            ('url_content_cache_data', self.url_content_cache_db.data_path),
            ('url_content_cache_index', self.url_content_cache_db.index_path),
            ('cache_info', self.cache_info_file)
        ]:
            if file_path.exists():
                stat = file_path.stat()
                stats['files'][name] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': stat.st_mtime
                }
            else:
                stats['files'][name] = {'exists': False}

        return stats