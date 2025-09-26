# Agentic RAG System for CVPR Research Comprehension

This repository contains the source code and documentation for a locally-hosted, agentic Retrieval-Augmented Generation (RAG) system. This project is designed to assist with the comprehension and reasoning over a corpus of Computer Vision and Pattern Recognition (CVPR) research papers from arXiv.

The system leverages open-source, fine-tuned language models and a vector database to provide intelligent, context-aware answers to user queries about complex academic literature.

## 📋 Project Overview

The core problem this project addresses is the difficulty of staying current with the rapid pace of academic research. This RAG system provides an intelligent interface to a library of CVPR papers, allowing users to ask questions, compare methodologies, and understand key contributions without manually sifting through dozens of documents.

The entire pipeline is built with open-source components and is designed to be executable on accessible hardware (like Kaggle's T4 GPUs), making it a powerful tool for students, researchers, and practitioners.

### ✨ Key Features

- **Automated Data Pipeline:** Scrapes metadata and full text of CVPR papers directly from arXiv.
- **Local LLM Fine-Tuning:** Utilizes QLoRA for memory-efficient fine-tuning of open-source language models (Qwen series) on domain-specific data.
- **Efficient Retrieval:** Employs a FAISS vector database with `sentence-transformer` embeddings for fast and accurate context retrieval.
- **Agentic Workflow:** Implements a LangChain agent that can reason and decide when to use the RAG pipeline as a tool to answer complex queries.
- **In-depth Evaluation:** Includes a comprehensive framework for evaluating both retriever and generator performance using metrics like Hit Rate, MRR, ROUGE, and Semantic Similarity.

## 🛠️ Architecture and Tech Stack

The system follows a modular RAG architecture, enabling independent development and upgrades of its components.

### System Flow

1. **Data Ingestion:** CVPR papers are scraped from arXiv.
2. **Text Processing:** Full text is extracted, cleaned, and segmented into chunks.
3. **Vectorization:** Text chunks are converted into semantic embeddings.
4. **Indexing:** Embeddings are stored and indexed in a FAISS vector database.
5. **RAG Process:**
   - A user's **Query** is vectorized.
   - FAISS **retrieves** the most relevant text chunks (context).
   - The query and context are passed to the fine-tuned **LLM**.
   - The LLM **generates** a factually grounded answer.

### ⚙️ Tech Stack

- **Core ML/DL:** `PyTorch`, `Transformers`, `PEFT`, `bitsandbytes`, `TRL`
- **RAG & Agents:** `LangChain`, `LangChain-Community`
- **Data Processing:** `arxiv`, `pymupdf`, `datasets`, `pandas`
- **Vector Database:** `faiss-gpu`, `sentence-transformers`
- **Visualization:** `networkx`, `matplotlib`

## 🚀 Setup and Usage

Follow these steps to set up and run the project in a suitable environment (e.g., Kaggle Notebook with 2x T4 GPUs).

### 1. Installation

Clone the repository and install the required dependencies.

```bash
git clone https://your-repo-url.git
cd your-repo-folder
pip install -r requirements.txt
```

*Note: A requirements.txt file should be created listing all packages from the initial !pip install cell in the notebook.*

### 2. Running the Pipeline

The project is structured as a series of steps, typically run sequentially in a Jupyter Notebook.

#### Step 1: Data Collection and Vector Store Creation
Execute the first script to scrape data from arXiv, process the PDFs, and create the `faiss_index_cvpr` vector store.

#### Step 2: Model Fine-Tuning
Run the fine-tuning script. You can specify the base model to be fine-tuned by changing the `model_id` variable. This step will produce a directory (e.g., `./qwen3-4b-finetuned`) containing the trained adapter layers.

#### Step 3: Merge Adapters
Run the merge script to combine the trained adapters with the base model. This creates a complete, standalone model directory (e.g., `./qwen3-4b-merged`) ready for inference.

#### Step 4: Querying the RAG Pipeline
Execute the RAG pipeline script, which loads the merged model and the FAISS index to answer queries.

## 🧠 Model Evolution and Performance Analysis

The project's core is its fine-tuned LLM. The system evolved through three distinct model iterations, with performance improving at each stage.

### Fine-Tuning Methodology: QLoRA

QLoRA (Quantized Low-Rank Adaptation) was used for all fine-tuning due to its efficiency. It allows for adapting large models on limited hardware by quantizing the model to 4-bits and training small adapter layers.

### Iteration 1: 1.5B Parameter Baseline (Qwen-1.5B)
- **Fine-Tuning Data:** A minimalist dataset created using only paper abstracts.
- **Analysis:** This model served as a crucial diagnostic tool. It suffered from severe "prompt leakage" and a high degree of hallucination. It was not reliable for Q&A.

### Iteration 2: 4B Parameter Upgrade (Qwen-1.5B)
- **Fine-Tuning Data:** Same abstracts-only dataset.
- **Analysis:** The increased model scale resolved the prompt leakage issue, and the model provided its first correct answers. However, performance was inconsistent, and it would often hallucinate when the retriever failed.

### Iteration 3: The Enriched 4B Model (Domain Expert)
- **Fine-Tuning Data:** A dramatically enriched dataset using all available metadata (authors, dates, categories, DOI, text previews, etc.). This forced the model to learn the paper's full context and focus on factual details.
- **Analysis:** This model showed a remarkable improvement. Its most significant capability was reasoned abstention—it would correctly identify when the context was insufficient and state that the answer was not available, rather than hallucinating.

### Quantitative Performance Comparison

The following table compares the quantitative metrics across the model iterations, showing a clear trajectory of improvement.

| Metric | 1.5B Model (Iter. 1) | 4B Model (Iter. 2) | Enriched 4B Model (Iter. 3) |
|--------|----------------------|---------------------|------------------------------|
| Retriever: Hit Rate | 50.00% | 50.00% | 85.71% |
| Retriever: MRR | 0.1583 | 0.1583 | 0.6714 |
| Generator: ROUGE-L | ~0.0410 | ~0.0350 | 0.0485 |
| Generator: Semantic Sim. | 0.5628 | 0.5759 | 0.6331 |

**Retriever Analysis:** The significant jump in Hit Rate and MRR in the final iteration is due to a more targeted evaluation dataset that better matched the enriched model's capabilities.

**Generator Analysis:** The key indicator of improvement is Semantic Similarity, which steadily increased from 0.5628 to a strong 0.6331, confirming the final model's superior contextual understanding.

## 🌐 Langflow Integration & Workflow Analysis

An attempt was made to integrate the final enriched 4B model into a visual, interactive workflow using Langflow. The integration was not successful due to a NoneType return value issue across the FAISS Loader and subsequent chain components, even though the return types were configured correctly. However, the process provided valuable insights.

### Component Configuration

Each individual component of the Langflow diagram was configured correctly and validated in isolation:

- **Ollama LLM Node:** Successfully connected to the locally hosted `final_qwen3_4b_cvpr` model.
- **HuggingFace Embeddings Node:** Correctly configured with the `all-MiniLM-L6-v2` model.
- **FAISS Loader Node:** Correctly loaded the index from the specified local path when tested independently.
- **PromptTemplate and Chain Nodes:** All other components were configured as expected.

### Root of the Integration Failure

Despite the correct individual configurations, the end-to-end flow failed to execute. The primary issue was a cascading error where key components would return a NoneType value instead of the expected object type (e.g., a Retriever object, a Chain object).

- **Cascading Failure:** The FAISS Loader, when integrated into the full flow, would return None instead of a valid Retriever object. This None value would then be passed to the RetrievalQA Chain node, causing it to fail its internal validation and also return None. This, in turn, caused the RAG Agent to fail.
- **Problem Analysis:** This behavior is indicative of an internal data passing or type validation issue within the Langflow framework for this specific combination of components. Even though the output and input types were correctly configured on the visual interface (e.g., Retriever -> Retriever), the underlying execution was not passing the object reference correctly. This suggests a potential bug, version incompatibility between components, or a subtle misconfiguration that is not exposed through the user interface.

## ⚠️ Project Limitations

The outcomes of this project were achieved under several key constraints:

- **Hardware:** Development on Kaggle's free-tier NVIDIA T4 GPUs was the primary limitation, restricting the size of models that could be fine-tuned and causing out-of-memory errors in the agentic workflow.
- **Model Scale:** The project was limited to models in the 1.5B-4B parameter range. Access to more powerful models (e.g., 7B+) would likely yield even stronger results.
- **Data Scale:** The fine-tuning dataset was based on a small corpus of 10 papers. A larger and more diverse dataset would significantly enhance the model's expertise.

## 🔮 Future Work

- **Enhance the Retriever:** Experiment with more powerful embedding models to further improve the Hit Rate and MRR.
- **Curated Fine-Tuning Dataset:** Create a larger, human-curated dataset to improve the model's reasoning and reduce reliance on synthetic data.
- **Explore Larger Models:** Re-run the pipeline on more powerful hardware (e.g., A100/H100 GPUs) using larger open-source models.
- **Resolve Langflow Integration:** Debug the interoperability issue to create a fully functional visual interface for the RAG agent.
