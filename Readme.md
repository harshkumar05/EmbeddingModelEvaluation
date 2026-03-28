# LLMEvaluation - Embedding Model Evaluation Framework

A comprehensive Python framework for evaluating and analyzing embedding models using multiple benchmarks, fairness tests, geometric properties, and retrieval metrics. Designed to work with local embedding models via Ollama.

**License:** MIT (Copyright © 2025 Harsh)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Documentation](#documentation)

---

## 🎯 Overview

This framework provides a suite of evaluation tools for embedding models, enabling you to:
- Test embedding space **isotropy** (distribution properties)
- Measure **semantic similarity** against human judgments (STS-Benchmark)
- Evaluate **word similarity** (SimLex-999)
- Assess **fairness** and bias (ECT, WEAT)
- Evaluate **information retrieval** performance (NDCG)
- Analyze geometric properties (alignment, uniformity, isotropy)
- Visualize embedding distributions

---

## 📁 Project Structure

```
LLMEvaluation/
├── tests/
│   ├── isotropyScore.py                 # Isotropy diagnostic tests
│   ├── geometry/
│   │   ├── Isotropy.py                  # Isotropy analysis with STS data
│   │   ├── Alignment.py                 # Alignment metric computation
│   │   ├── AlignmentAndUniformity.py    # Combined geometric metrics
│   │   ├── VisualizeEmbedding.py        # Embedding visualization
│   │   └── ot.py                        # Optimal transport utilities
│   ├── benchmarking/
│   │   ├── Sts.py                       # STS-Benchmark evaluation
│   │   ├── Wms.py                       # Word Meaning Similarity (SimLex-999)
│   │   └── somecheck.py                 # Miscellaneous checks
│   ├── fairness/
│   │   ├── Ect.py                       # Embedding Coherence Test
│   │   └── Weat.py                      # Word Embedding Association Test
│   └── retrival/
│       └── Ndgc.py                      # NDCG - Information Retrieval evaluation
├── projectUtils/
│   ├── __init__.py
│   └── ollamaUtils.py                   # Ollama API wrapper utilities
├── datasets/
│   └── benchmarking/
│       └── SimLex-999.txt               # SimLex-999 dataset
├── references/
│   └── benchmarkings/
│       ├── stsBenchmarking.txt          # STS benchmark results
│       └── simlexBenchmarking.txt       # SimLex benchmark results
├── pictures/                             # Visualization outputs
│   ├── embeddingGemmaIsotroySacn.png
│   ├── NdgcMrrScan.png
│   ├── stsScan.png
│   ├── wmsScan.png
│   └── Ect.png
├── setup.py                             # Package configuration
├── requirements.txt                     # Python dependencies
├── ollamaTest.py                        # Quick test script
└── Readme.md                            # This file
```

---

## ✨ Features

### 1. **Geometry Tests**
- **Isotropy Analysis**: Detect anisotropy (cone problem) in embedding spaces
  - Narrow Cone Test (average cosine similarity)
  - PCA variance analysis
  - Dimensionality collapse detection
- **Alignment Metrics**: Measure semantic alignment in paired embeddings
- **Uniformity**: Check if embeddings are uniformly distributed
- **Visualization**: 2D/3D visualization of embedding distributions

### 2. **Benchmarking**
- **STS-Benchmark**: Evaluate contextual semantic similarity (Spearman correlation)
- **Word Meaning Similarity (SimLex-999)**: Test word-level similarity against human ratings

### 3. **Fairness & Bias Assessment**
- **ECT (Embedding Coherence Test)**: Measure bias between gender/attribute pairs
- **WEAT (Word Embedding Association Test)**: Quantify implicit associations in embeddings

### 4. **Information Retrieval**
- **NDCG Metrics**: Evaluate retrieval performance with traps and challenges
  - Lexical traps
  - Negation traps
  - Syntactic traps
  - Polysemy traps

### 5. **Utility Functions**
- Safe Ollama API integration with batching
- Error handling and timeout support
- Embedding normalization and verification

---

## 📋 Prerequisites

- **Python 3.9+**
- **Ollama** with Embedding Gemma model (or compatible embedding model)
- **pip** package manager

### Install Ollama

Download and install from: https://ollama.com/download

### Pull Embedding Model

```bash
ollama pull embedding-gemma:latest
```

Or use any compatible embedding model (e.g., `nomic-embed-text`, `mxbai-embed-large`).

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone git@github.com:harshkumar05/LLMEvaluation.git
cd LLMEvaluation
```

### Step 2: Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Ollama Connection

```bash
python ollamaTest.py
```

---

## 🎬 Quick Start

### Example 1: Test Isotropy

```python
from tests.geometry import Isotropy

# Isotropy will load STS data and analyze embedding isotropy
isotropy_test = Isotropy()
```

### Example 2: Run STS Benchmark

```bash
python tests/benchmarking/Sts.py
```

This evaluates sentence similarity using Spearman correlation.

### Example 3: Check Word Similarity (SimLex-999)

```bash
python tests/benchmarking/Wms.py
```

### Example 4: Fairness Assessment (ECT)

```bash
python tests/fairness/Ect.py
```

### Example 5: Information Retrieval (NDCG)

```bash
python tests/retrival/Ndgc.py
```

---

## 📚 Test Categories

### Geometry Tests (`tests/geometry/`)

| Test | Purpose | Metric |
|------|---------|--------|
| **Isotropy** | Detect cone problem in embedding space | Avg cosine similarity, PCA variance |
| **Alignment** | Measure semantic alignment in pairs | Alignment score |
| **AlignmentAndUniformity** | Combined geometric analysis | Alignment + uniformity scores |
| **VisualizeEmbedding** | Visualize embeddings in 2D/3D | PCA/TSNE plots |

### Benchmarking Tests (`tests/benchmarking/`)

| Test | Dataset | Metric |
|------|---------|--------|
| **STS** | STS-Benchmark (sentence pairs) | Spearman correlation |
| **Wms** | SimLex-999 (word pairs) | Spearman correlation |

### Fairness Tests (`tests/fairness/`)

| Test | Purpose | Method |
|------|---------|--------|
| **ECT** | Embedding Coherence Test | Gender/attribute bias measurement |
| **WEAT** | Word Embedding Association Test | Implicit association detection |

### Retrieval Tests (`tests/retrival/`)

| Test | Purpose | Metric |
|------|---------|--------|
| **NDCG** | Information retrieval ranking | NDCG@k score |

---

## 📖 Key Dependencies

- **ollama** - Local embedding model API
- **numpy** - Numerical computations
- **scikit-learn** - ML utilities (PCA, normalization, metrics)
- **scipy** - Statistical analysis (Spearman correlation)
- **pandas** - Data manipulation
- **matplotlib** - Visualization
- **datasets** - HuggingFace datasets (STS, GLUE benchmarks)
- **openai** - Optional: For API-based models
- **google-genai** - Optional: For Google embedding APIs

---

## 🔧 Utility Functions

### Ollama Integration (`projectUtils/ollamaUtils.py`)

```python
from projectUtils.ollamaUtils import (
    get_embedding,              # Single embedding
    get_embeddings_batch,       # Batch embeddings
    get_embeddings_batch_via_api  # Via HTTP API
)

# Example
embeddings = get_embeddings_batch('embedding-gemma:latest', sentences)
```

---

## 📊 Output & Results

Results are saved in:
- `references/benchmarkings/` - Benchmark comparison data
- `pictures/` - Visualization outputs
- Console - Real-time evaluation metrics

### Example Output Metrics

```
=== STS-BENCHMARK EVALUATION ===
Spearman Correlation: 0.8541 ✅
P-value: < 0.001

=== ISOTROPY ANALYSIS ===
Avg Cosine Similarity: 0.3254 ✅ (Healthy)
Top-3 PCA Variance: 42.15% ✅

=== FAIRNESS TEST ===
Male-Tech Association: 0.65
Female-Tech Association: 0.42
Bias Score: 0.23 ⚠️ (Moderate)
```

---

## 🐛 Troubleshooting

### Issue: Connection refused (Ollama)
```
Solution: Ensure Ollama is running
$ ollama serve
```

### Issue: Model not found
```
Solution: Pull the model first
$ ollama pull embedding-gemma:latest
```

### Issue: Timeout errors
```
Solution: Increase timeout in ollamaUtils.py (default: 120s)
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact & Support

For questions or issues, please open a GitHub issue or contact the maintainer.

---

## 🎓 References

- **STS-Benchmark**: https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
- **SimLex-999**: https://fh295.github.io/simlex.html
- **WEAT**: https://arxiv.org/abs/1608.08187
- **Isotropy**: https://arxiv.org/abs/1910.10341
- **NDCG**: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

---

**Last Updated:** March 2026
    

    


