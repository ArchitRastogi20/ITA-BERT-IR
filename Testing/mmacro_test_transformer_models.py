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
import psutil
from typing import List, Generator, Dict, Any
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data files
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('contriever_pooling_experiments.txt', mode='w')

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

# Initialize tokenizer and Contriever model
try:
    logger.info("Loading Facebook Contriever model and tokenizer...")
    tokenizer_contriever = AutoTokenizer.from_pretrained("facebook/mcontriever-msmarco")
    contriever_model = AutoModel.from_pretrained("facebook/mcontriever-msmarco").to(device)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    sys.exit(1)

# Load stopwords
try:
    logger.info("Loading stopwords...")
    italian_stopwords = set(stopwords.words('italian'))
    logger.info(f"Loaded {len(italian_stopwords)} Italian stopwords")
except Exception as e:
    logger.error(f"Error loading stopwords: {e}")
    italian_stopwords = set()  # Fallback to empty set
    
# Text preprocessing function
def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Preprocess text by tokenizing and optionally removing stopwords
    """
    if not remove_stopwords:
        return text
    
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in italian_stopwords]
    return " ".join(filtered_tokens)

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

# Function to compute embeddings with different pooling strategies
def compute_embeddings(
    model: AutoModel,
    texts: List[str],
    tokenizer: AutoTokenizer,
    pooling_strategy: str = 'cls',
    batch_size: int = 128,
    num_workers: int = 8
) -> np.ndarray:
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Computing {pooling_strategy} embeddings"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Apply different pooling strategies
            if pooling_strategy == 'cls':
                # CLS token pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling_strategy == 'mean':
                # Mean pooling (considering attention mask)
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            elif pooling_strategy == 'max':
                # Max pooling (with attention mask)
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                embeddings = outputs.last_hidden_state * attention_mask_expanded
                # Replace padding with large negative value
                embeddings = embeddings.masked_fill((1 - attention_mask_expanded).bool(), -1e9)
                batch_embeddings = torch.max(embeddings, dim=1)[0].cpu().numpy()
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                
            embeddings_list.append(batch_embeddings)
            
            # Log memory usage
            mem_usage = get_memory_usage()
            logger.debug(f"Current memory usage: {mem_usage:.2f} GB")
    
    all_embeddings = np.vstack(embeddings_list)
    return all_embeddings

# Dynamic batch size adjustment
def adjust_batch_size(initial_batch_size: int) -> int:
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_memory < 2:
        return max(1, initial_batch_size // 4)
    elif available_memory < 4:
        return max(1, initial_batch_size // 2)
    else:
        return initial_batch_size

# Document and query text lists
texts = df_collection['text'].tolist()
doc_ids = df_collection['id'].astype(str).tolist()
queries = df_dataset['query'].tolist()
positive_doc_ids = df_dataset['id_doc_collection'].astype(str).tolist()  # Ground truth doc IDs

batch_size = adjust_batch_size(128)
logger.info(f"Adjusted batch size: {batch_size}")

# Configure experiments
pooling_strategies = ['cls', 'mean', 'max']
stopwords_options = [False, True]  # False = keep stopwords, True = remove stopwords

# Evaluation parameters
K_R_list = [1, 5, 100]  # For Recall@K
K_AP = max(K_R_list)  # For Average Precision
K_NDCG_list = [10, 5]  # For NDCG@K
K_MRR_list = [10, 100, 1000]  # For MRR@K
logger.info(f"Evaluation parameters: K_R_list={K_R_list}, K_AP={K_AP}, K_NDCG_list={K_NDCG_list}, K_MRR_list={K_MRR_list}")

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

# Function to retrieve top K documents using cosine similarity
def retrieve_top_k(query_embeddings, doc_embeddings, doc_ids, top_k):
    retrieved_ids_list = []
    for query_emb in tqdm(query_embeddings, desc="Retrieving documents"):
        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, query_emb)
        # Get top K indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        # Get corresponding document IDs
        retrieved_ids = [doc_ids[idx] for idx in top_k_indices]
        retrieved_ids_list.append(retrieved_ids)
    return retrieved_ids_list

# Create a summary table for results
results = []

# Run experiments for all combinations
for remove_stopwords in stopwords_options:
    # Preprocess texts
    logger.info(f"Preprocessing texts (remove_stopwords={remove_stopwords})...")
    processed_texts = [preprocess_text(text, remove_stopwords) for text in tqdm(texts, desc="Processing documents")]
    processed_queries = [preprocess_text(query, remove_stopwords) for query in tqdm(queries, desc="Processing queries")]
    
    for pooling in pooling_strategies:
        experiment_name = f"contriever_{pooling}_stopwords_{remove_stopwords}"
        logger.info(f"\n{'='*20} Running experiment: {experiment_name} {'='*20}")
        
        # Set up ChromaDB collection
        try:
            logger.info(f"Setting up ChromaDB collection for {experiment_name}...")
            client = chromadb.Client(Settings())
            if experiment_name in [col.name for col in client.list_collections()]:
                client.delete_collection(experiment_name)
            collection = client.create_collection(experiment_name)
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {e}")
            continue
        
        # Compute document embeddings
        try:
            logger.info(f"Computing document embeddings (pooling={pooling}, remove_stopwords={remove_stopwords})...")
            start_time = time.time()
            doc_embeddings = compute_embeddings(
                contriever_model, 
                processed_texts, 
                tokenizer_contriever, 
                pooling_strategy=pooling,
                batch_size=batch_size
            )
            logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
            
            # Add document embeddings to ChromaDB
            logger.info("Adding embeddings to ChromaDB...")
            for i in tqdm(range(0, len(processed_texts), 1000), desc="Adding embeddings"):
                batch_texts = processed_texts[i:i+1000]
                batch_embeddings = doc_embeddings[i:i+1000].tolist()
                batch_ids = doc_ids[i:i+1000]
                collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    ids=batch_ids
                )
        except Exception as e:
            logger.error(f"Error computing document embeddings: {e}")
            continue
        
        # Compute query embeddings
        try:
            logger.info(f"Computing query embeddings (pooling={pooling}, remove_stopwords={remove_stopwords})...")
            start_time = time.time()
            query_embeddings = compute_embeddings(
                contriever_model, 
                processed_queries, 
                tokenizer_contriever, 
                pooling_strategy=pooling,
                batch_size=batch_size
            )
            logger.info(f"Query embeddings shape: {query_embeddings.shape}")
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error computing query embeddings: {e}")
            continue
        
        # Perform retrieval and evaluate
        logger.info("Retrieving documents and computing metrics...")
        max_top_k = max(max(K_R_list), max(K_NDCG_list), max(K_MRR_list))
        retrieved_doc_ids_list = retrieve_top_k(
            query_embeddings, doc_embeddings, doc_ids, top_k=max_top_k
        )
        
        R_at_K, average_precision, NDCG_at_K, MRR_at_K = compute_metrics(
            retrieved_doc_ids_list, positive_doc_ids, K_R_list, K_AP, K_NDCG_list, K_MRR_list
        )
        
        # Log results
        logger.info(f"\n{'='*20} Results for {experiment_name} {'='*20}")
        for K in K_R_list:
            logger.info(f"R@{K}: {R_at_K[K]:.4f}")
        logger.info(f"Average Precision: {average_precision:.4f}")
        for K in K_NDCG_list:
            logger.info(f"NDCG@{K}: {NDCG_at_K[K]:.4f}")
        for K in K_MRR_list:
            logger.info(f"MRR@{K}: {MRR_at_K[K]:.4f}")
        
        # Store results for summary
        result_entry = {
            'Experiment': experiment_name,
            'Pooling': pooling,
            'Remove Stopwords': remove_stopwords,
            'Average Precision': average_precision
        }
        for K in K_R_list:
            result_entry[f'R@{K}'] = R_at_K[K]
        for K in K_NDCG_list:
            result_entry[f'NDCG@{K}'] = NDCG_at_K[K]
        for K in K_MRR_list:
            result_entry[f'MRR@{K}'] = MRR_at_K[K]
        
        results.append(result_entry)

# Create summary dataframe
summary_df = pd.DataFrame(results)
logger.info("\n=========== SUMMARY OF ALL EXPERIMENTS ===========")
logger.info(summary_df.to_string())

# Save results to CSV
csv_filename = 'contriever_pooling_stopwords_results.csv'
summary_df.to_csv(csv_filename, index=False)
logger.info(f"Results saved to {csv_filename}")

logger.info("\nExperiment completed successfully!")