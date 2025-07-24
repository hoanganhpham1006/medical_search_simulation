# Guide for Using Multiple Embedding Sources

## Overview
To use embeddings from multiple sources, you'll need to manually combine them into a single folder and ensure the metadata dataset contains information for all sources.

## Steps to Combine Multiple Sources

### 1. Prepare Your Embedding Files
If you have embeddings from multiple sources, you'll need to:

1. Copy all embedding files into a single folder
2. Rename files to avoid conflicts (ensure sequential numbering)
3. Update MAX_EMBEDDING_FILES in config.py to reflect the total number of files

Example:
```bash
# If you have:
# - Source 1: embeddings_0.pt to embeddings_999.pt
# - Source 2: embeddings_0.pt to embeddings_499.pt

# Copy and rename:
cp /path/to/source1/embeddings_*.pt /path/to/combined/
cd /path/to/combined/
# Rename source2 files to continue from source1
for i in {0..499}; do
  cp /path/to/source2/embeddings_$i.pt embeddings_$((1000+i)).pt
done

# Update config.py:
# MAX_EMBEDDING_FILES = 1500  # Total files from all sources
```

### 2. Combine Metadata Datasets
Your metadata dataset should contain entries for all papers/documents from all sources. The dataset should have:
- `passage_id`: List of passage IDs that map to embedding indices
- `paper_id`: Unique identifier for each paper
- `paper_title`: Title of the paper
- `paper_url`: URL of the paper
- `passage_text`: List of passage texts
- Other metadata fields (year, venue, specialty, etc.)

### 3. Update Configuration
Edit `config.py`:
```python
# Update to point to combined folder
EMBEDDING_FOLDER = "/path/to/combined/embeddings/"
MAX_EMBEDDING_FILES = 1500  # Total number of files

# Update to combined metadata dataset
METADATA_DATASET_NAME = "your/combined_metadata_dataset"
```

### 4. Clear Existing Index
Remove the old FAISS index so a new one is built:
```bash
rm /mnt/sharefs/tuenv/wiki_search_cache/faiss_index.bin
```

### 5. Run the API
The system will automatically build a new FAISS index from all combined embeddings:
```bash
python api.py
```

## Important Notes

1. **Passage ID Mapping**: Ensure that passage IDs in the metadata correspond correctly to the embedding file indices after combining.

2. **Memory Requirements**: Combining multiple sources will increase memory usage during index building.

3. **Index Building Time**: The first run will take longer as it needs to build the index from all embeddings.

4. **Consistency**: Make sure the embedding dimension is the same across all sources (configured as EMBEDDING_DIMENSION in config.py).