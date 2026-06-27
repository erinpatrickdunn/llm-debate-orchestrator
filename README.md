
---

# LLM-Debate-Orchestrator

An automated, local multi-agent debate sandbox powered by **Ollama**.

The **LLM-Debate-Orchestrator** is a specialized research tool designed to facilitate adversarial academic debates between AI agents. By utilizing an asymmetric memory-slicing architecture and local Retrieval-Augmented Generation (RAG), this framework allows users to stress-test the argumentative consistency, logical progression, and empirical grounding of various LLM models in a closed-loop, local environment.

## 🖥️ Hardware Recommendations

Because this orchestrator runs models and vector embeddings locally, performance depends on your hardware. These requirements ensure stable inference and avoid latency issues.

* **OS:** Linux Mint 22.3 (64-bit Mate Edition) recommended.
* **CPU:** Ryzen 7 2700X or equivalent.
* **RAM:** 16GB DDR4 minimum.
* **Storage:** M.2 NVMe SSD (highly recommended for rapid model loading).
* **GPU:** 8GB VRAM (minimum for stable performance on 7B+ parameter models).

> **Note:** If running models larger than 4B parameters, 8GB of VRAM is recommended. If no dedicated GPU is available, ensure at least 16GB of system RAM to allow for CPU-based inference.
> 
> 

## 🛠️ Infrastructure & Setup

### 1. Prerequisites

* [Ollama](https://ollama.com/) must be installed and running.
* Python 3.10+ installed.

### 2. Environment Initialization

Clone the repository and initialize the secure sandbox:

```bash
# Initialize the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

```

### 3. Launching

Start the environment control board:

```bash
python llm-debate-orchestrator.py

```

## 🚀 Key Features

* **Infrastructure Dependency Board:** A real-time diagnostic dashboard that verifies your environment health before any simulation begins.
* **Asymmetric Memory Slicing:** Agents maintain distinct personas governed by `custom_prompts/`, preventing context repetition during long-form debates.
* **Zombie Process Protection:** The script manages the lifecycle of the Ollama daemon. When you exit through the Main Menu (Option 4), the orchestrator automatically identifies and terminates any spawned background processes, preventing "zombie" services.
* **Adversarial RAG Pipeline:** Automatically chunks and vectorizes documents in `rag_data/`, ensuring that agents are grounded in your specific research files rather than just hallucinating context.

## 🧩 Architectural Flow

1. **Ingestion:** Text files in `rag_data/` are processed into a vector space for semantic retrieval.
2. **Orchestration:** The script queries the vector store to ground Agent A’s opening thesis.
3. **Adversarial Loop:** Outputs are passed between Agent A (Proponent) and Agent B (Skeptic) for a specified number of rounds.
4. **Judicial Verdict:** The full transcript is synthesized by the "Axiom" judge model, producing a final verdict in `axiom_verdict.txt`.

## ⚠️ Limitations

* **Resource Intensive:** Local LLM inference is VRAM-heavy. Monitor your system resources during multi-turn debates.
* **Stochastic Variance:** Using different models (e.g., `qwen2.5:7b` vs `llama3`) will lead to different rhetorical outcomes.
* **Persona Sensitivity:** Debater behavior is highly dependent on the system prompts located in `/custom_prompts/`.

## 📜 Repository Structure

* `/custom_prompts/`: System instructions for debaters and the judicial judge.
* `/rag_data/`: Place source documents (.txt, .md) here for grounded inference.
* `llm-debate-orchestrator.py`: The core controller and menu logic.
* `requirements.txt`: Python package requirements (requests, numpy).

---

### Pro-Tip

Use the **Markdown PDF** extension in VS Code to export the `axiom_verdict.txt` or your final transcripts into professional PDF documents for your research files.
