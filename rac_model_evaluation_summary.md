## 🧪 Model Evaluation on Subset of 200 Records (116 Categories)

This subset contains 200 records from the original DBpedia dataset, covering 116 categories out of 219 total. The goal is to compare the performance of three classification approaches: **KNN**, **LLM-only (BART)**, and the hybrid **RAC** model.

### 📊 Results Summary

| Model        | Accuracy | Precision | F1-Score |
|--------------|----------|-----------|----------|
| **KNN**      | 0.20     | 0.13      | 0.13     |
| **LLM-only** | 0.20     | 0.11      | 0.12     |
| **RAC**      | **0.50** | **0.32**  | **0.33** |

---

### 📌 Observations

#### 🔹 KNN:
- Weak performance due to high category sparsity.
- The semantic similarity search fails when nearest neighbors are not semantically relevant.
- Sensitive to imbalanced training data.

#### 🔹 LLM-only (Zero-Shot Classification with `facebook/bart-large-mnli`):
- Accuracy is similar to KNN, but precision and F1 are low.
- LLM struggles with a large number of unfamiliar candidate labels.
- Without retrieval context, the model's predictions are often inaccurate.

#### 🔹 RAC (Retrieval-Augmented Classification):
- Best overall performance.
- Combines top-k semantic retrieval with few-shot examples for in-context classification.
- Reduces label confusion by limiting prediction candidates to those retrieved by KNN.

---

### ⚠️ Sklearn Warning
Some warnings such as:

```
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
```

occur because some categories in the test set were never predicted. To handle this, we used:
```python
precision_score(..., zero_division=0)
```

---

### ✅ Conclusion

- RAC significantly improves classification quality over both KNN and zero-shot LLM.
- Prompt engineering and category filtering based on nearest neighbors are key to RAC's success.
- This hybrid method is especially effective when dealing with many classes and limited training data per class.
