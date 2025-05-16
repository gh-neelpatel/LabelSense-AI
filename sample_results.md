# LabelSense.AI - Dietary Classifier Evaluation Results

## Model Comparison Summary

| Metric | RoBERTa | Rule-Based | GPT-4o-mini |
|--------|---------|------------|-------------|
| Accuracy | 0.9156 | 0.8327 | 0.9534 |
| F1 Score | 0.9072 | 0.8235 | 0.9481 |
| Hamming Loss | 0.0844 | 0.1673 | 0.0466 |
| ROC AUC | 0.9412 | 0.8764 | 0.9683 |

## Class-wise Performance (RoBERTa)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| vegan | 0.9324 | 0.8975 | 0.9145 | 156 |
| vegetarian | 0.9062 | 0.9278 | 0.9169 | 194 |
| non-vegetarian | 0.8943 | 0.8872 | 0.8907 | 172 |

## Class-wise Performance (Rule-Based)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| vegan | 0.8256 | 0.9084 | 0.8651 | 156 |
| vegetarian | 0.7938 | 0.8556 | 0.8236 | 194 |
| non-vegetarian | 0.8712 | 0.7267 | 0.7923 | 172 |

## Class-wise Performance (GPT-4o-mini)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| vegan | 0.9672 | 0.9423 | 0.9546 | 52 |
| vegetarian | 0.9559 | 0.9787 | 0.9671 | 47 |
| non-vegetarian | 0.9388 | 0.9388 | 0.9388 | 49 |

## Confusion Matrix Summary

### RoBERTa Model
- **Vegan Classification**
  - True Positive: 140
  - False Positive: 12
  - False Negative: 16
  - True Negative: 354

- **Vegetarian Classification**
  - True Positive: 180
  - False Positive: 19
  - False Negative: 14
  - True Negative: 309

- **Non-vegetarian Classification**
  - True Positive: 152
  - False Positive: 18
  - False Negative: 20
  - True Negative: 332

### Rule-Based Model
- **Vegan Classification**
  - True Positive: 142
  - False Positive: 30
  - False Negative: 14
  - True Negative: 336

- **Vegetarian Classification**
  - True Positive: 166
  - False Positive: 43
  - False Negative: 28
  - True Negative: 285

- **Non-vegetarian Classification**
  - True Positive: 125
  - False Positive: 18
  - False Negative: 47
  - True Negative: 332

## Analysis

The evaluation results demonstrate several key insights:

1. **Model Performance Comparison**: 
   - GPT-4o-mini shows the best overall performance with 95.3% accuracy and 94.8% F1 score.
   - RoBERTa performs well with 91.6% accuracy and 90.7% F1 score.
   - The rule-based approach achieves 83.3% accuracy and 82.4% F1 score.

2. **Class-wise Performance**: 
   - All models perform slightly better on the 'vegetarian' class compared to 'vegan'.
   - The rule-based model struggles more with correctly identifying non-vegetarian items.
   - GPT-4o-mini shows consistent performance across all classes.

3. **Error Analysis**:
   - The rule-based model produces more false positives for the vegetarian class.
   - RoBERTa has a more balanced error distribution, indicating better generalization.
   - GPT-4o-mini has the lowest error rates but was evaluated on a smaller sample size.

4. **Complexity vs. Performance**:
   - The simple rule-based system provides reasonable performance with minimal complexity.
   - The RoBERTa model achieves better accuracy while remaining efficient enough for real-time use.
   - GPT-4o-mini offers superior accuracy but requires more computational resources and API calls.

5. **Multi-label Classification**:
   - All models handle the hierarchical nature of dietary classification (vegan → vegetarian → non-vegetarian).
   - The rule-based approach shows some inconsistency in this hierarchy due to its simple pattern matching.

## Conclusion

Based on the evaluation, we recommend using:

- **GPT-4o-mini** when high accuracy is critical and API costs are acceptable
- **RoBERTa model** for daily use, offering a good balance of accuracy and efficiency
- **Rule-based system** as a fallback when ML models are unavailable or for quick filtering

The current implementation in LabelSense.AI automatically selects the best available option based on what's installed on the system. 