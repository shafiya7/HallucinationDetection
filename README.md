# Graph-Based Hallucination Detection in Large Language Models

## Overview

This project implements a graph-based relational reasoning framework for hallucination detection in large language models. The objective is to determine whether a generated claim is factually consistent or hallucinated by modeling structured relationships between claims, evidence, and entities.

Two approaches are implemented and compared:

- Flat similarity-based baseline using Sentence-BERT embeddings and cosine similarity
- Graph-based relational model using a heterogeneous multi-relational graph and Relational Graph Convolutional Networks (RGCN)

The model is trained on the FEVER dataset and evaluated on both structured verification data and the TruthfulQA benchmark to assess generalization performance.

---

## Repository Contents

- hallucination_detection.ipynb — Complete end-to-end implementation
- README.md — Project documentation

All preprocessing, model training, validation, testing, and visualization steps are contained within the notebook.

---

## System Requirements

- Python 3.9 or higher
- pip package manager
- Internet connection for dataset downloads
- Minimum 8 GB RAM
- Recommended: GPU-enabled environment for faster training

The model can run on CPU, but GPU significantly reduces training time.

---

## Required Python Packages

Install the following dependencies before running the notebook:

pip install torch torchvision torchaudio  
pip install torch-geometric  
pip install sentence-transformers  
pip install scikit-learn  
pip install spacy  
pip install datasets  
python -m spacy download en_core_web_sm  

If using GPU, ensure CUDA-compatible PyTorch is installed.

---

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Set runtime to GPU (optional but recommended).
3. Run all cells sequentially from top to bottom.
4. Final evaluation results will be printed at the end of execution.

### Option 2: Local Jupyter Notebook

1. Install required dependencies.
2. Open the notebook in Jupyter.
3. Run all cells sequentially.
4. Ensure internet access for dataset loading.

---

## Method Summary

### Dataset

Training Dataset:
- FEVER dataset
- SUPPORTED mapped to non-hallucinated (0)
- REFUTED mapped to hallucinated (1)
- NOT ENOUGH INFO samples excluded
- 70% training, 15% validation, 15% testing split

Evaluation Dataset:
- TruthfulQA validation split
- Used for cross-domain generalization testing

---

### Flat Similarity-Based Model

- Sentence-BERT embeddings (all-MiniLM-L6-v2)
- Embedding dimension: 384
- Cosine similarity between claim and evidence
- Threshold-based hallucination classification
- No relational modeling

---

### Graph-Based Relational Model

Graph Construction:
- Nodes: claim, evidence, entity nodes
- Edges: claim–evidence, evidence–claim, claim–entity, evidence–entity
- Maximum 3 evidence sentences per claim for efficiency

Model Architecture:
- Two-layer Relational Graph Convolutional Network (RGCN)
- Hidden dimension: 256
- Dropout: 0.3
- ReLU activation
- Sigmoid output layer for binary classification

Training Configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Weighted Binary Cross Entropy loss
- Model selection based on validation F1 score

---

## Evaluation Metrics

The following metrics are computed:

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under ROC Curve (AUROC)

F1 score is emphasized due to class imbalance.

---

## Reproducibility

To reproduce results:

1. Install all required dependencies.
2. Run the notebook from the first cell to the last.
3. Do not skip installation or preprocessing steps.
4. Ensure stable internet connection for dataset downloads.

All hyperparameters are explicitly defined in the notebook. Results may vary slightly due to random initialization.

---

## Limitations

- TruthfulQA evaluation uses simplified evidence construction.
- External retrieval module is not integrated.
- Graph size is limited for computational efficiency.
- Cross-domain performance may vary.

---

## Future Work

Future work will focus on enhancing relational reasoning and generalization capabilities. Integrating retrieval-based evidence construction would allow the model to ground claims using external knowledge sources. Incorporating natural language inference for relation typing may improve modeling of support and contradiction signals within the graph. Extending the framework to dynamically construct graphs for multi-sentence and long-form responses could strengthen real-world applicability. Evaluating the approach on larger-scale hallucination benchmarks and exploring more efficient graph architectures may further improve robustness and scalability.

---

This repository is structured to ensure reproducibility, clarity, and alignment with the methodology described in the final IEEE report.
