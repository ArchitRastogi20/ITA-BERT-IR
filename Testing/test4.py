import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import time
import logging
import sys
import os
import psutil  # For memory usage
from typing import List, Generator, Dict, Any

# Additional imports for BM25
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data files (if not already downloaded)
nltk.download('punkt', quiet=True)

# Set up logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('script_output_mul_reacll_ap_bdcg_mrr.txt', mode='w')

c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
if not logger.hasHandlers():
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Function to get current memory usage
def get_memory_usage() -> float:
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # Convert bytes to GB
    return mem

# Load data
try:
    logger.info("Loading datasets...")
    df_dataset = pd.read_csv('df_dataset.csv')
    df_collection = pd.read_csv('df_collection.csv')
    logger.info(f"Number of queries: {len(df_dataset)}")
    logger.info(f"Number of collection documents: {len(df_collection)}")
except FileNotFoundError as e:
    logger.error(f"Dataset file not found: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    sys.exit(1)

# Initialize tokenizer and models
base_model_name = 'dbmdz/bert-base-italian-xxl-uncased'  # Replace with the actual base model name
finetuned_model_name = 'finetuned_model'  # Replace with your finetuned model path

try:
    logger.info("Loading tokenizer and models...")
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name).to(device)
    finetuned_model = AutoModel.from_pretrained(finetuned_model_name).to(device)

    # Load Contriever model
    tokenizer_contriever = AutoTokenizer.from_pretrained("facebook/mcontriever-msmarco")
    contriever_model = AutoModel.from_pretrained("facebook/mcontriever-msmarco").to(device)

    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    sys.exit(1)

# Define dataset class
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

# Function to compute embeddings using a generator
def compute_embeddings_generator(
    model: AutoModel,
    texts: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 128,
    num_workers: int = 8
) -> Generator[np.ndarray, None, None]:
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            # Memory-efficient computation
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            yield embeddings

            # Log memory usage
            mem_usage = get_memory_usage()
            logger.debug(f"Current memory usage: {mem_usage:.2f} GB")

# Function to compute all embeddings and collect them
def compute_all_embeddings(
    model: AutoModel,
    texts: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 128,
    num_workers: int = 8
) -> np.ndarray:
    embeddings_list = []
    for embeddings in compute_embeddings_generator(model, texts, tokenizer, batch_size, num_workers):
        embeddings_list.append(embeddings)
    all_embeddings = np.vstack(embeddings_list)
    return all_embeddings

# Dynamic batch size adjustment based on available memory
def adjust_batch_size(initial_batch_size: int) -> int:
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_memory < 2:
        return max(1, initial_batch_size // 4)
    elif available_memory < 4:
        return max(1, initial_batch_size // 2)
    else:
        return initial_batch_size

# Compute embeddings for collection documents
texts = df_collection['text'].tolist()
doc_ids = df_collection['id'].astype(str).tolist()

batch_size = adjust_batch_size(128)
logger.info(f"Adjusted batch size: {batch_size}")

try:
    logger.info("Computing base model embeddings for collection...")
    start_time = time.time()
    base_embeddings = compute_all_embeddings(base_model, texts, tokenizer_base, batch_size=batch_size, num_workers=8)
    logger.info(f"Base embeddings shape: {base_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing base embeddings: {e}")
    sys.exit(1)

try:
    logger.info("Computing finetuned model embeddings for collection...")
    start_time = time.time()
    finetuned_embeddings = compute_all_embeddings(finetuned_model, texts, tokenizer_base, batch_size=batch_size, num_workers=8)
    logger.info(f"Finetuned embeddings shape: {finetuned_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing finetuned embeddings: {e}")
    sys.exit(1)

try:
    logger.info("Computing Contriever embeddings for collection...")
    start_time = time.time()
    contriever_embeddings = compute_all_embeddings(contriever_model, texts, tokenizer_contriever, batch_size=batch_size, num_workers=8)
    logger.info(f"Contriever embeddings shape: {contriever_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing Contriever embeddings: {e}")
    sys.exit(1)

# Initialize ChromaDB client
logger.info("Initializing ChromaDB client...")
client = chromadb.Client(Settings())
logger.info("ChromaDB client initialized.")

# Create collections
logger.info("Setting up collections in ChromaDB...")
# Delete existing collections if they exist
existing_collections = [col.name for col in client.list_collections()]
for col_name in ["base_embeddings", "finetuned_embeddings", "contriever_embeddings"]:
    if col_name in existing_collections:
        client.delete_collection(col_name)

base_collection = client.create_collection("base_embeddings")
finetuned_collection = client.create_collection("finetuned_embeddings")
contriever_collection = client.create_collection("contriever_embeddings")
logger.info("Collections are set up.")

# Add embeddings to collections in batches
def add_embeddings_in_batches(
    collection: chromadb.api.models.Collection,
    texts: List[str],
    embeddings: np.ndarray,
    ids: List[str],
    batch_size: int = 10000
):
    num_entries = len(texts)
    for i in tqdm(range(0, num_entries, batch_size), desc="Adding embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size].tolist()
        batch_ids = ids[i:i+batch_size]
        try:
            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                ids=batch_ids
            )
        except Exception as e:
            logger.error(f"Error adding embeddings to collection: {e}")
            continue  # Continue with the next batch

logger.info("Adding base model embeddings to ChromaDB...")
add_embeddings_in_batches(base_collection, texts, base_embeddings, doc_ids)
logger.info("Base model embeddings added.")

logger.info("Adding finetuned model embeddings to ChromaDB...")
add_embeddings_in_batches(finetuned_collection, texts, finetuned_embeddings, doc_ids)
logger.info("Finetuned model embeddings added.")

logger.info("Adding Contriever embeddings to ChromaDB...")
add_embeddings_in_batches(contriever_collection, texts, contriever_embeddings, doc_ids)
logger.info("Contriever embeddings added.")

# Compute embeddings for queries
queries = df_dataset['query'].tolist()
positive_doc_ids = df_dataset['id_doc_collection'].astype(str).tolist()  # Ground truth doc IDs

try:
    logger.info("Computing base model embeddings for queries...")
    start_time = time.time()
    base_query_embeddings = compute_all_embeddings(base_model, queries, tokenizer_base, batch_size=batch_size, num_workers=8)
    logger.info(f"Base query embeddings shape: {base_query_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing base query embeddings: {e}")
    sys.exit(1)

try:
    logger.info("Computing finetuned model embeddings for queries...")
    start_time = time.time()
    finetuned_query_embeddings = compute_all_embeddings(finetuned_model, queries, tokenizer_base, batch_size=batch_size, num_workers=8)
    logger.info(f"Finetuned query embeddings shape: {finetuned_query_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing finetuned query embeddings: {e}")
    sys.exit(1)

try:
    logger.info("Computing Contriever embeddings for queries...")
    start_time = time.time()
    contriever_query_embeddings = compute_all_embeddings(contriever_model, queries, tokenizer_contriever, batch_size=batch_size, num_workers=8)
    logger.info(f"Contriever query embeddings shape: {contriever_query_embeddings.shape}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Error computing Contriever query embeddings: {e}")
    sys.exit(1)

# Evaluation parameters with configurable K values
K_R_list = [1, 100, 1000]          # For R@1, R@100, R@1000
K_AP = max(K_R_list)               # For Average Precision
K_NDCG_list = [10, 100, 1000]      # For NDCG@10, NDCG@100, NDCG@1000
K_MRR_list = [10, 100, 1000]       # For MRR@10, MRR@100, MRR@1000
num_queries = len(queries)
logger.info(f"Evaluation parameters: K_R_list={K_R_list}, K_AP={K_AP}, K_NDCG_list={K_NDCG_list}, K_MRR_list={K_MRR_list}")

###########################################
# BM25 Implementation with NDCG and MRR
###########################################

logger.info("Preparing BM25 retrieval...")

# Tokenize collection documents
tokenized_corpus = [word_tokenize(doc.lower()) for doc in tqdm(texts, desc="Tokenizing collection")]
bm25 = BM25Okapi(tokenized_corpus)

# Tokenize queries
tokenized_queries = [word_tokenize(query.lower()) for query in tqdm(queries, desc="Tokenizing queries")]

# Perform BM25 retrieval
logger.info("Performing BM25 retrieval...")
bm25_retrieved_doc_ids_list = []
for query_tokens in tqdm(tokenized_queries, desc="BM25 retrieval"):
    doc_scores = bm25.get_scores(query_tokens)
    top_n_indices = np.argsort(doc_scores)[::-1][:max(K_NDCG_list + K_MRR_list)]
    retrieved_ids = [doc_ids[idx] for idx in top_n_indices]
    bm25_retrieved_doc_ids_list.append(retrieved_ids)

# Function to compute metrics
def compute_metrics(retrieved_doc_ids_list, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list):
    num_queries = len(positive_doc_ids)
    R_at_K = {K: 0 for K in K_R_list}
    AP_total = 0.0
    NDCG_at_K = {K: 0.0 for K in K_NDCG_list}
    MRR_at_K = {K: 0.0 for K in K_MRR_list}
    for i in range(num_queries):
        ground_truth_doc_id = positive_doc_ids[i]
        retrieved_doc_ids = retrieved_doc_ids_list[i]

        # Compute R@K for each K
        for K in K_R_list:
            if ground_truth_doc_id in retrieved_doc_ids[:K]:
                R_at_K[K] +=1

        # Compute Average Precision
        try:
            rank = retrieved_doc_ids[:K_AP].index(ground_truth_doc_id) + 1
            AP = 1.0 / rank
            AP_total += AP
        except ValueError:
            pass  # Ground truth not in top K_AP

        # Compute NDCG@K for each K
        for K in K_NDCG_list:
            DCG = 0.0
            for j, doc_id in enumerate(retrieved_doc_ids[:K], start=1):
                rel = 1.0 if doc_id == ground_truth_doc_id else 0.0
                if rel > 0:
                    DCG += (2 ** rel - 1) / np.log2(j + 1)
                    break  # Since we have only one relevant document
            # Compute IDCG
            IDCG = (2 ** 1 - 1) / np.log2(1 + 1)
            NDCG = DCG / IDCG if IDCG > 0 else 0.0
            NDCG_at_K[K] += NDCG

        # Compute MRR@K for each K
        for K in K_MRR_list:
            try:
                rank = retrieved_doc_ids[:K].index(ground_truth_doc_id) + 1
                RR = 1.0 / rank
                MRR_at_K[K] += RR
            except ValueError:
                pass  # Ground truth not in top K

    # Compute averages
    R_at_K = {K: R_at_K[K]/num_queries for K in K_R_list}
    average_precision = AP_total / num_queries
    NDCG_at_K = {K: NDCG_at_K[K] / num_queries for K in K_NDCG_list}
    MRR_at_K = {K: MRR_at_K[K] / num_queries for K in K_MRR_list}

    return R_at_K, average_precision, NDCG_at_K, MRR_at_K

# Evaluate BM25
logger.info("Computing metrics for BM25...")
R_at_K_bm25, average_precision_bm25, NDCG_at_K_bm25, MRR_at_K_bm25 = compute_metrics(
    bm25_retrieved_doc_ids_list, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list
)

###########################################
# Evaluate base model using cosine similarity with NDCG and MRR
###########################################

logger.info("Evaluating base model...")

# Function to compute cosine similarity and retrieve top K
def retrieve_top_k(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    top_k: int
) -> List[List[str]]:
    retrieved_ids_list = []
    for query_emb in tqdm(query_embeddings, desc="Retrieving documents"):
        similarities = np.dot(doc_embeddings, query_emb)
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_ids = [doc_ids[idx] for idx in top_k_indices]
        retrieved_ids_list.append(retrieved_ids)
    return retrieved_ids_list

# Perform retrieval using cosine similarity
max_top_k = max(K_NDCG_list + K_MRR_list)
logger.info("Retrieving documents using base model embeddings...")
retrieved_doc_ids_list_base = retrieve_top_k(
    base_query_embeddings, base_embeddings, doc_ids, top_k=max_top_k
)

# Evaluate base model
logger.info("Computing metrics for base model...")
R_at_K_base, average_precision_base, NDCG_at_K_base, MRR_at_K_base = compute_metrics(
    retrieved_doc_ids_list_base, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list
)

###########################################
# Evaluate finetuned model using cosine similarity with NDCG and MRR
###########################################

logger.info("Evaluating finetuned model...")

# Perform retrieval using cosine similarity
logger.info("Retrieving documents using finetuned model embeddings...")
retrieved_doc_ids_list_finetuned = retrieve_top_k(
    finetuned_query_embeddings, finetuned_embeddings, doc_ids, top_k=max_top_k
)

# Evaluate finetuned model
logger.info("Computing metrics for finetuned model...")
R_at_K_finetuned, average_precision_finetuned, NDCG_at_K_finetuned, MRR_at_K_finetuned = compute_metrics(
    retrieved_doc_ids_list_finetuned, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list
)

###########################################
# Evaluate Contriever model using cosine similarity with NDCG and MRR
###########################################

logger.info("Evaluating Contriever model...")

# Perform retrieval using cosine similarity
logger.info("Retrieving documents using Contriever model embeddings...")
retrieved_doc_ids_list_contriever = retrieve_top_k(
    contriever_query_embeddings, contriever_embeddings, doc_ids, top_k=max_top_k
)

# Evaluate Contriever model
logger.info("Computing metrics for Contriever model...")
R_at_K_contriever, average_precision_contriever, NDCG_at_K_contriever, MRR_at_K_contriever = compute_metrics(
    retrieved_doc_ids_list_contriever, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list
)

###########################################
# Summary of Results
###########################################

logger.info("\n=========== Summary of Results ===========")
# BM25 Results
logger.info("BM25 Results:")
for K in K_R_list:
    logger.info(f"R@{K}: {R_at_K_bm25[K]:.4f}")
logger.info(f"Average Precision: {average_precision_bm25:.4f}")
for K in K_NDCG_list:
    logger.info(f"NDCG@{K}: {NDCG_at_K_bm25[K]:.4f}")
for K in K_MRR_list:
    logger.info(f"MRR@{K}: {MRR_at_K_bm25[K]:.4f}")

# Base Model Results
logger.info("\nBase Model(dbmdz/bert-base-italian-xxl-uncased) Results:")
for K in K_R_list:
    logger.info(f"R@{K}: {R_at_K_base[K]:.4f}")
logger.info(f"Average Precision: {average_precision_base:.4f}")
for K in K_NDCG_list:
    logger.info(f"NDCG@{K}: {NDCG_at_K_base[K]:.4f}")
for K in K_MRR_list:
    logger.info(f"MRR@{K}: {MRR_at_K_base[K]:.4f}")

# Finetuned Model Results
logger.info("\nFinetuned Model Results:")
for K in K_R_list:
    logger.info(f"R@{K}: {R_at_K_finetuned[K]:.4f}")
logger.info(f"Average Precision: {average_precision_finetuned:.4f}")
for K in K_NDCG_list:
    logger.info(f"NDCG@{K}: {NDCG_at_K_finetuned[K]:.4f}")
for K in K_MRR_list:
    logger.info(f"MRR@{K}: {MRR_at_K_finetuned[K]:.4f}")

# Contriever Model Results
logger.info("\nfacebook/mcontriever-msmarco Model Results:")
for K in K_R_list:
    logger.info(f"R@{K}: {R_at_K_contriever[K]:.4f}")
logger.info(f"Average Precision: {average_precision_contriever:.4f}")
for K in K_NDCG_list:
    logger.info(f"NDCG@{K}: {NDCG_at_K_contriever[K]:.4f}")
for K in K_MRR_list:
    logger.info(f"MRR@{K}: {MRR_at_K_contriever[K]:.4f}")

logger.info("==========================================")
