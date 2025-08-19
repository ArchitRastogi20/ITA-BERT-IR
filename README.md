

# ITA-BERT-IR — Italian Embeddings for Information Retrieval

Fine-tuning **`dbmdz/bert-base-italian-xxl-uncased`** to produce high-quality **Italian sentence/document embeddings** for retrieval tasks.
The published model is available on Hugging Face:

* **Model card:** [https://huggingface.co/ArchitRastogi/bert-base-italian-embeddings](https://huggingface.co/ArchitRastogi/bert-base-italian-embeddings)

This repo also includes quick benchmarking utilities to compare the finetuned model against the base model and a sparse BM25 baseline on an Italian IR setup (queries + collection). The evaluation scripts compute Recall\@K and MRR, and support different pooling strategies and a simple ChromaDB-based retriever. 

---

## Contents

```
ITA-BERT-IR/
├── Train/
│   ├── fix_dataset.py                 # utilities to clean/convert data for training
│   └── train_cont.py                  # contrastive / embedding training script
├── Testing/
│   ├── mMacro_testing.py              # compare base vs finetuned model + BM25, logs R@K/MRR
│   ├── mmacro_test_transformer_models.py
│   │                                  # generic transformer embedding testbed (pooling, ChromaDB)
│   ├── df_dataset.csv                 # sample queries file (example)
│   └── df_collection.csv              # sample collection file (example)
└── LICENSE                            # Apache-2.0
```

> The evaluation scripts explicitly set `base_model_name = "dbmdz/bert-base-italian-xxl-uncased"` and `finetuned_model_name = "ArchitRastogi/bert-base-italian-embeddings"`, and log metrics such as R\@K and MRR. 
> The generic transformer test harness loads a pretrained model/tokenizer, computes embeddings with CLS/mean/max pooling, and can index/query using ChromaDB. 

---

## Why this repo?

* **Native Italian embeddings:** tuned starting from an Italian BERT to better capture Italian semantics for retrieval.
* **Simple, reproducible eval:** quick scripts to compare **base vs finetuned** and a **BM25** baseline on an Italian query–document setup. 
* **Plug-and-play inference:** easy to load from Hugging Face and drop into RAG or search systems.

---

## Quick start

### 1) Environment

Python ≥ 3.9 is recommended.

```bash
# Create & activate a virtual env (example)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install core deps
pip install torch transformers chromadb nltk pandas numpy tqdm psutil scikit-learn
```

> The testing utilities rely on `transformers`, `torch`, `nltk`, `pandas`, `numpy`, `tqdm`, `psutil`, and optionally `chromadb` for light-weight nearest-neighbor search. 

### 2) Data layout for evaluation

Put two CSVs under `Testing/` (examples already included):

* `df_collection.csv`: your **document collection**. Typical columns:

  * `doc_id` (string/int) — unique id
  * `text` (string) — document text
* `df_dataset.csv`: your **query set**. Typical columns:

  * `query` (string) — query text
  * `relevant_id` (string/int) — id from `doc_id` that’s considered relevant

> If your schema differs, open the testing script and adapt the column names in the data-loading section. The scripts are simple and designed to be edited.

---

## Run evaluation

### A) Compare base vs finetuned + BM25

This script loads both the base Italian BERT and your finetuned model, runs embedding retrieval, and reports **Recall\@K** and **MRR**; it also logs a BM25 baseline for reference. ([GitHub][1])

```bash
python Testing/mMacro_testing.py
```

What it does (high level):

* Loads tokenizer/models:

  * Base: `dbmdz/bert-base-italian-xxl-uncased`
  * Finetuned: `ArchitRastogi/bert-base-italian-embeddings`
* Creates embeddings for collection and queries (configurable pooling).
* Retrieves top-K via cosine similarity; computes R\@K and MRR.
* Prints a summary for each model and the sparse baseline. ([GitHub][1])

### B) Generic embedding testbed (pooling, batch size, ChromaDB)

For controlled experiments (pooling strategy, dynamic batching, memory logging, etc.), use:

```bash
python Testing/mmacro_test_transformer_models.py
```

Key features:

* CLS / mean / max pooling over transformer outputs
* Adjustable batch size with memory-aware scaling
* Optional ChromaDB indexing & retrieval for the collection
* Logs throughput and memory usage to console/file ([GitHub][2])

---

## Training

> The training utilities live under `Train/`. Use `fix_dataset.py` to convert/clean your raw data and `train_cont.py` to run contrastive/embedding training.

Typical workflow:

1. **Prepare training data**

   ```bash
   python Train/fix_dataset.py \
     --input /path/to/raw.csv \
     --output /path/to/processed.csv
   ```

   (Adjust arguments to your data; open the script to see expected columns/options.)

2. **Run training**

   ```bash
   python Train/train_cont.py \
     --base_model dbmdz/bert-base-italian-xxl-uncased \
     --train_file /path/to/processed.csv \
     --output_dir ./outputs/italian-embeddings \
     --epochs 1 --batch_size 32 --lr 2e-5
   ```

   > Exact CLI flags may differ; check the script header / `argparse` section and tune to your resources/dataset.

3. **(Optional) Push to Hugging Face**

   After training, log in with `huggingface-cli login` and push the model to your namespace.

---

## Inference (use the published model)

Load the embeddings model directly from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F

MODEL_NAME = "ArchitRastogi/bert-base-italian-embeddings"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def embed_texts(texts, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device).eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            out = model(**enc).last_hidden_state
            vecs = mean_pool(out, enc["attention_mask"])
            vecs = F.normalize(vecs, p=2, dim=1)  # L2-normalize for cosine similarity
            all_vecs.append(vecs.cpu())
    return torch.cat(all_vecs, dim=0)

queries   = ["come rinnovare la carta d’identità", "migliori ristoranti a Trastevere"]
docs      = ["La carta d’identità si rinnova in comune...", "Elenco dei ristoranti…", "I passaggi per il passaporto…"]

q_emb = embed_texts(queries)
d_emb = embed_texts(docs)

# cosine similarity
sims = (q_emb @ d_emb.T)
topk = sims.topk(k=2, dim=1).indices
print(topk)   # indices of top matches in docs for each query
```

---

## Tips & notes

* **Pooling:** start with **mean pooling**; CLS may underperform on retrieval tasks. The test scripts let you try both and **max pooling**. 
* **Normalization:** L2-normalize embeddings before similarity.
* **Batching:** tune batch size based on GPU RAM; the test harness includes a simple memory-aware adjustment. 
* **Baselines:** always compare against BM25 for sanity (included in the testing script). 


---

## License

This project is released under the **Apache-2.0** License (see `LICENSE`).

---

## Acknowledgements / References

* Base model: **dbmdz/bert-base-italian-xxl-uncased**
* Finetuned model: **ArchitRastogi/bert-base-italian-embeddings**
* Evaluation scaffolding in `Testing/` uses standard `transformers`, optional **ChromaDB**, and logs R\@K/MRR. 

---

