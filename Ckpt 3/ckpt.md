# LabelSense: Evaluation & Model Summary

## 1. Model & Approach Decision Matrix

| Component       | Approach     | Reason                       | Alternative Approach         |
|------------------------|------------------------|------------------------------------------|------------------------------|
| **OCR**                | EasyOCR                | Open-source, multilingual, fast          | Google OCR |
| **Ingredient NER**     | BERT-NER               | Structured, high F1, adaptable           | GPT-4, dictionary lookup     |
| **Dietary Classification** | RoBERTa-base       | High accuracy, multi-label               | GPT-4, LightGBM              |
| **Allergen Detection** | Hybrid: Rule + GPT-4   | Rules for explicit, LLM for inferred     | N/A    |
| **Ingredient Explanation** | GPT-4 + Caching   | Clear, human-like, simple                | Wikipedia/API      |
| **Personalization**    | Rules + GPT            | Deterministic + flexible fallback        | DistilBERT |
| **Evaluation**    | Quantitative + Survey  | Captures performance + user feedback     | N/A|

---

## 2. Evaluation Metrics Matrix

| Component               | Primary Metric(s)             | Reason                                      | Secondary Metrics       | Human Evaluation?              |
|-------------------------|-------------------------------|------------------------------------------|-------------------------|--------------------------|
| **OCR**                 | CER, WER                      | Text accuracy vs. ground truth           | BLEU, Levenshtein       | No                       |
| **Ingredient NER**      | Macro F1-score                | Balance precision/recall                 | Precision, Recall       | No                       |
| **Dietary Classification** | Multi-label Acc., F1      | Handles multi-tag, class imbalance       | Hamming Loss, ROC-AUC   | Trust Score (1–5)        |
| **Allergen Detection**  | Recall                        | Safety-critical, catch-all               | F1, Precision           | Trust Score (1–5)        |
| **Ingredient Explanation** | User Score (1–5)           | Clarity and comprehension                | BLEU, ROUGE             | Pairwise Ranking         |
| **Personalization**     | Recall (Profile-based)        | Matches individual user needs            | False Negative Rate     | Scenario Test            |
| **System UX**           | SUS, Task Time                | Ease of use, trust, overall satisfaction | Qualitative Feedback    | SUS Survey               |

---

## 3. Human Feedback Analysis Methods

| Analysis Type   | Action                  | Tool / Output           | Purpose                                        |
|-----------------|-------------------------|-------------------------|------------------------------------------------|
| **Quantitative**   | Calculate mean/std      | Pandas, Excel           | Summarize ratings (trust, SUS)                 |
| **Quantitative**   | Visualize scores        | Matplotlib, seaborn     | Compare models or user groups                  |
| **Quantitative**   | Correlate scores        | Spearman, Pearson       | Link metrics to system performance             |
| **Qualitative**    | Code open feedback      | Manual, Taguette        | Identify recurring themes in free-text feedback|
| **Qualitative**    | Highlight key quotes    | Markdown/LaTeX          | Support qualitative claims                     |
| **Qualitative**    | Generate word clouds    | Python (wordcloud lib), online tools | Visualize frequent terms           |
