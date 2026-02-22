
# NOTES YT: AN AI-DRIVEN RAG FRAMEWORK FOR AUTOMATED VIDEO SUMMARIZATION AND SEMANTIC KNOWLEDGE RETRIEVAL

**Keywords:** NLP, LLMs, RAG, Vector Databases, Transformers, Summarization, Semantic Search, Chunking, Embeddings, Streamlit, Python, Inference Latency, BERTScore, ROUGE, Orchestration, Quantization, Prompt Engineering.

---

## 1. Abstract

The exponential growth of educational content on YouTube has created a "content overload" challenge. While video is a rich medium, it is inherently linear and time-consuming to consume. **Notes YT** is an AI-driven system designed to transform this passive experience into an active learning process. The system utilizes a multi-stage NLP pipeline to automate transcript extraction, perform intelligent text chunking, and generate abstractive summaries using Large Language Models (LLMs) like **Flan-T5**.

The final version implements a **Retrieval-Augmented Generation (RAG)** framework. By integrating a **Vector Database**, the system indexes video content as high-dimensional embeddings, allowing for semantic search and precise question-answering. Initial benchmarking on a **Google Colab T4 GPU** demonstrates high semantic accuracy, with a **BERTScore of 0.94**, validating the system's ability to preserve core meaning while reducing consumption time by over 80%.

## 2. Problem Statement

The current digital landscape presents several critical challenges for efficient video-based knowledge management:

* **Time Inefficiency:** Users spend excessive time watching long-form videos to find specific information.
* **Linear Constraints:** Video lacks native "searchability" for internal concepts.
* **Context Window Limits:** Standard LLMs fail when processing transcripts from multi-hour lectures due to token limits.
* **Lack of Persistence:** Existing tools rarely offer cohesive environments to save notes or interaction history.
* **Information Retrieval Gaps:** Simple keyword searches fail to capture semantic meaning.

## 3. Objectives

* **Automate Extraction:** Programmatically fetch and clean transcripts from diverse YouTube URL formats.
* **Context-Aware Summarization:** Convert raw transcripts into structured bullet-point notes using LLMs.
* **Intelligent Chunking:** Design a sliding window strategy to overcome model token limits.
* **Semantic Interactivity (RAG):** Integrate a vector database for context-grounded Q&A.
* **Model Flexibility:** Allow users to select between different LLM architectures.
* **Persistence:** Implement methods to save notes and chat histories.

## 4. Scope

* **Platform:** Specifically optimized for YouTube (standard, shortened, and mobile URLs).
* **Processing:** Handles videos with manual or auto-generated transcripts.
* **Models:** Implementation of the **Flan-T5** family with modular support for other Hugging Face models.
* **RAG Architecture:** Use of **FAISS** or **ChromaDB** for high-dimensional vector storage.
* **Deployment:** Optimized for local execution, Google Colab, and Streamlit Cloud.

## 5. Literature Review

The project is grounded in several seminal advancements:

* **Transformers:** Vaswani et al. (2017) introduced the self-attention mechanism, allowing for superior handling of long-range dependencies in text.
* **T5 & Flan-T5:** Raffel et al. (2019) and Chung et al. (2022) demonstrated the power of unified text-to-text models and instruction tuning for high-quality zero-shot performance.
* **RAG:** Lewis et al. (2020) formalized the combination of generative LLMs with external non-parametric memory to reduce hallucinations.
* **Vector Search:** Research on **FAISS** (Johnson et al., 2017) provided the infrastructure for millisecond-latency semantic retrieval.

## 6. Methodology

1. **Ingestion:** Capture URL and retrieve raw transcript snippets via the YouTube Transcript API.
2. **Chunking:** Divide the corpus into 1,200-character segments to fit within model context windows.
3. **Summarization:** Pass chunks to Flan-T5 using **Beam Search** to generate coherent, abstractive notes.
4. **RAG Implementation:** Generate text embeddings and store them in a Vector Database for similarity-based retrieval.
5. **Interface:** Wrap the logic in a Streamlit framework using **Session State** for memory and history.

## 7. Proposed System

The system follows a three-tier architecture:

* **Presentation Layer:** Streamlit interface for user input and chat rendering.
* **Application Layer:** Python-based orchestrator managing data flow and error handling.
* **AI & Data Layer:** The inference engine (Flan-T5) and the Vector Engine for semantic indexing.

## 8. Expected Results

Initial testing on a **Google Colab T4 instance** yielded the following:

* **System Metrics:** Average transcript fetch in **1.12s**; LLM inference in **6.86s**; Total pipeline in **7.99s**.
* **Accuracy Metrics:** **BERTScore F1 of 0.9449** (Excellent semantic alignment); **ROUGE-1 of 0.5545** (Strong concept retention).
* **Impact:** Users can expect an 80%+ reduction in information consumption time.

## 9. Tools and Technologies

* **Language:** Python
* **Frontend:** Streamlit
* **AI Frameworks:** Hugging Face Transformers, PyTorch, Sentence-Transformers
* **LLM:** Google Flan-T5
* **APIs:** YouTube Transcript API
* **Vector Storage:** ChromaDB / FAISS
* **Hardware:** NVIDIA T4 GPU (for benchmarking)

## 10. References

* **Vaswani, A., et al. (2017).** *Attention Is All You Need*.
* **Raffel, C., et al. (2019).** *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*.
* **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
* **Chung, H. W., et al. (2022).** *Scaling Instruction-Finetuned Language Models*.
* **Kharwal, A. (2026).** *Build an AI System to Summarize YouTube Videos into Notes*. AmanXai.
