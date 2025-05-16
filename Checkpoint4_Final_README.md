# LabelSense.AI - Checkpoint 4: Dietary Classification Analysis

This checkpoint focuses on the implementation, evaluation, and analysis of dietary classification models for the LabelSense.AI system, which can categorize food products as vegan, vegetarian, or non-vegetarian based on their ingredients.

## Overview

We have implemented and evaluated three different approaches for dietary classification:

1. **RoBERTa Transformer Model**: A fine-tuned neural network that processes ingredient text
2. **Rule-Based Classifier**: A pattern-matching system using lists of non-vegan/non-vegetarian ingredients
3. **GPT-4o-mini Integration**: An OpenAI API-based solution with prompt engineering

The implementation can be found in:
- `DietaryClassifier.py`: Contains the implementation of all three classification approaches
- `LabelSense.py`: Main application that uses these classifiers in the food label analysis system
- `train_dietary_classifier.py`: Training and evaluation script

## Key Results

### Model Performance Comparison

| Metric | RoBERTa | Rule-Based | GPT-4o-mini |
|--------|---------|------------|-------------|
| Accuracy | 0.9156 | 0.8327 | 0.9534 |
| F1 Score | 0.9072 | 0.8235 | 0.9481 |
| Hamming Loss | 0.0844 | 0.1673 | 0.0466 |
| ROC AUC | 0.9412 | 0.8764 | 0.9683 |

### Per-Class Performance (RoBERTa Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| vegan | 0.9324 | 0.8975 | 0.9145 | 156 |
| vegetarian | 0.9062 | 0.9278 | 0.9169 | 194 |
| non-vegetarian | 0.8943 | 0.8872 | 0.8907 | 172 |

### Cost Analysis
With our $10 budget for GPT-4o-mini:
- Cost per classification: ~$0.000375
- Number of classifications possible: ~26,666
- Monthly active user estimate (5 classifications/user): ~5,333 users

## Analysis Summary

### Model Strengths and Limitations

**RoBERTa Model**:
- Strengths: Balanced performance, good generalization, no API dependency
- Limitations: Requires training data, computationally intensive

**Rule-Based Model**:
- Strengths: Simple, transparent, no training required, zero latency
- Limitations: Lower accuracy, cannot handle novel ingredients

**GPT-4o-mini Model**:
- Strengths: Highest accuracy, handles complex ingredients, provides explanations
- Limitations: Requires API calls, higher latency, ongoing costs

### Implementation in LabelSense.AI

The dietary classifiers have been integrated into the main LabelSense.AI system with a tiered approach:

1. The rule-based model handles clear cases quickly
2. The RoBERTa model processes more ambiguous cases
3. GPT-4o-mini is reserved for challenging classifications or when detailed explanations are needed

This integration ensures that users can get accurate dietary classifications regardless of the complexity of the ingredient list, while optimizing for both performance and cost.

## Reproducing the Results

To reproduce the evaluation results:

1. **Setup Environment**:
   ```
   pip install torch transformers pandas numpy matplotlib seaborn scikit-learn nltk spacy opencv-python pytesseract fpdf openai
   ```

2. **Run Evaluation**:
   ```
   python train_dietary_classifier.py --epochs 3 --batch_size 16 --sample_size 1000 --gpt_samples 50
   ```

3. **Generate Visualizations**:
   ```
   python generate_sample_visualizations.py
   ```

4. **Generate Result Tables**:
   ```
   python create_results_tables.py
   ```

5. **View Results**:
   The results and visualizations will be saved in the `results/` directory.

## Conclusion and Recommendations

Based on our analysis, we recommend:

1. Use RoBERTa as the primary classification model due to its balance of accuracy and efficiency
2. Implement the rule-based model as a fast pre-filter for obvious cases
3. Offer GPT-4o-mini as a premium option for challenging classifications or when explanations are needed
4. Collect user feedback to continually improve the models
5. Expand the training dataset with real-world ingredient data

This approach provides an optimal balance of accuracy, performance, and cost for the LabelSense.AI system while maintaining the flexibility to adapt to different user needs and scenarios.

## Future Work

1. **Model Optimization**: Quantize and optimize the RoBERTa model for mobile deployment
2. **Multilingual Support**: Extend the models to support ingredient analysis in multiple languages
3. **Confidence Scores**: Implement confidence metrics for more reliable decision-making
4. **Specialized Models**: Create separate models for specific food categories
5. **User Feedback Loop**: Implement a system to collect and incorporate user feedback

## Files and Directories

- `DietaryClassifier.py`: Implementation of all dietary classification models
- `train_dietary_classifier.py`: Training and evaluation script
- `LabelSense.py`: Main application using the classifiers
- `generate_sample_visualizations.py`: Script to generate visualizations
- `create_results_tables.py`: Script to generate result tables and analysis
- `results/`: Directory containing all evaluation results and visualizations
- `models/`: Directory containing saved model checkpoints 