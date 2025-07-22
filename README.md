# RAC-Project

# 🧠 Text Classification with KNN, LLM, and Retrieval-Augmented Classification (RAC)

This project demonstrates a comparative approach to **text classification** using three different methods:
1. **K-Nearest Neighbors (KNN)**
2. **Large Language Models (LLM)**
3. **Retrieval-Augmented Classification (RAC)**

---

## 📁 Dataset

Start with a dataset (`.csv`) that contains:
- **Text samples**
- **Their corresponding categories**

The task is to **predict the correct category** for each text.

---

## 🔍 Approaches

### 1. KNN (K-Nearest Neighbors)
- Converts text samples into vector **embeddings**.
- Stores embeddings in **ChromaDB** (vector database).
- For each test sample, finds the **k most similar texts** from the training set.
- Predicts the **most common category** among those neighbors.

### 2. LLM (facebook/bart-large-mnli)
- Utilizes a **zero-shot classification pipeline** via HuggingFace.
- For each text, predicts the most likely category from all available options.
- Leverages **semantic understanding** from a large pretrained model.

### 3. RAC (Retrieval-Augmented Classification)
- **Combines KNN and LLM**:
  - First retrieves top-k neighbors (using KNN).
  - Then uses the **LLM**, but **only considers the categories** of the retrieved neighbors.
  - Optionally includes **neighbor texts as few-shot examples** in the prompt.
- This hybrid model benefits from both **retrieval-based context** and **LLM reasoning**.

---

## 🛠️ How the Code Works

### 1. Data Preparation
- Loads the dataset from CSV.
- Splits data into **training and testing sets**.
- Prepares samples for **embedding** and **classification**.

### 2. KNN
- Embeds training data using Sentence Transformers (e.g., `bge-small-en`).
- Stores embeddings in a **ChromaDB vector store**.
- Retrieves neighbors at prediction time and selects the most common category.

### 3. LLM
- Applies **facebook/bart-large-mnli** via HuggingFace's pipeline.
- Predicts category using **zero-shot inference** on full category list.

### 4. RAC
- Uses **KNN** to find nearest examples and categories.
- Prompts the **LLM** with:
  - The test text
  - Only the categories retrieved (making classification easier)
  - Optionally includes sample texts (few-shot style)

### 5. Evaluation
- Computes and prints:
  - ✅ Accuracy
  - ✅ Precision
  - ✅ F1 Score
- For each classification method (KNN, LLM, RAC)

---

## 🤔 Why This Approach?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **KNN** | Fast, uses semantic similarity, simple | No reasoning, depends on embedding quality |
| **LLM** | Powerful reasoning, good for unseen texts | May confuse similar categories, slower |
| **RAC** | Combines both – retrieval + reasoning | More complex implementation |

**RAC often performs better**, especially in low-data or long-tail scenarios.

---

## 📦 Requirements

- Python 3.8+
- `pandas`, `scikit-learn`, `transformers`, `torch`
- `chromadb`, `langchain`, `tqdm`

---

## 🚀 Run the Project

```bash
git clone https://github.com/monamahdavi/RAC-Project
cd RAC-Project
python LLM.py

