# Innovation Idea: Confidence-Guided Ingredient Knowledge Expansion

## Motivation

From our previous checkpoints, we observed that:
- The rule-based model struggles with ambiguous or novel ingredients not present in its keyword lists.
- RoBERTa improves on these cases but still fails on rare, regional, or context-dependent ingredients.
- GPT-4o-mini achieves the highest accuracy, especially on complex or ambiguous cases, but is costly and slower.

**Key finding:** Many misclassifications occur when the models encounter ingredients that are either missing from the rule-based lists or are poorly represented in the training data.

## Proposed Idea

### Confidence-Guided Ingredient Knowledge Expansion

1. **Confidence Thresholding:**  
   - For each classification, use the RoBERTa model’s confidence score (e.g., softmax probability).
   - If confidence is high, accept the prediction.
   - If confidence is low, escalate the sample to GPT-4o-mini for classification and explanation.

2. **Automated Ingredient List Expansion:**  
   - When GPT-4o-mini is used, extract its explanation (e.g., “seitan is a wheat-based vegan protein”).
   - Parse these explanations to identify new or ambiguous ingredients and their dietary status.
   - Periodically review and add these ingredients to the rule-based lists and/or augment the RoBERTa training data.

3. **Feedback Loop:**  
   - Maintain a log of escalated cases and their explanations.
   - Use this log to update both the rule-based and RoBERTa models, reducing future reliance on GPT-4o-mini.

## Effects of the Idea

| Effect Type | Description |
|-------------|-------------|
| **Positive** | Reduces cost and latency by limiting GPT-4o-mini calls to only uncertain cases. |
| **Positive** | Improves the rule-based and RoBERTa models over time by learning from real-world edge cases. |
| **Positive** | Expands ingredient coverage, especially for regional or novel terms, improving overall accuracy. |
| **Positive** | Provides more transparent and explainable classifications for users. |
| **Positive** | Creates a self-improving system that adapts to new data and user needs. |
| **Negative** | Requires careful parsing of GPT-4o-mini explanations to avoid introducing errors. |
| **Negative** | May need human review to validate new ingredient entries before updating lists. |
| **Negative** | Adds complexity to the system (confidence scoring, escalation, logging, parsing). |
| **Negative** | Risk of propagating GPT-4o-mini’s occasional mistakes if not properly filtered. |
| **Negative** | Initial implementation effort is higher due to the need for explanation extraction and feedback logic. |

## Why This Follows from Analysis

This idea directly addresses the observed weaknesses in the rule-based and RoBERTa models (edge cases, novel ingredients) and leverages the strengths of GPT-4o-mini (explanations, broad knowledge) in a cost-effective, scalable way. It is a logical next step based on the error analysis and model disagreement findings from previous checkpoints.

---