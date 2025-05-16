# LabelSense.AI - Checkpoint 4: Analysis

This checkpoint focuses on implementing and evaluating a dietary classification model for the LabelSense.AI system. The model can determine if a food product is vegan, vegetarian, or non-vegetarian based on its ingredients.

## Overview

The LabelSense.AI system has been extended with three different approaches for dietary classification:

1. **RoBERTa-base Model**: A fine-tuned transformer model for ingredient classification
2. **Rule-Based Classifier**: A simple pattern-matching approach using ingredient keywords
3. **GPT-4o-mini**: Leveraging OpenAI's model for zero-shot classification

## Directory Structure

```
LabelSense.AI/
├── LabelSense.py                    # Main application
├── DietaryClassifier.py             # Dietary classification models
├── train_dietary_classifier.py      # Training and evaluation script
├── models/                          # Directory for saved models
├── data/                            # Directory for datasets
├── results/                         # Results and visualizations
│   ├── dietary_classification_analysis.pdf    # PDF report
│   └── dietary_classification_results.json    # Raw results
└── Checkpoint4_README.md            # This file
```

## Setup

1. Install required packages:

```bash
pip install torch torchvision torchaudio transformers pandas numpy matplotlib seaborn scikit-learn tqdm fpdf pillow opencv-python pytesseract nltk spacy openai
```

2. Download required NLTK and spaCy resources:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

3. (Optional) Set your OpenAI API key in `train_dietary_classifier.py` to enable GPT-4o-mini evaluation.

## Running the Code

### Training and Evaluating Models

Run the following command to train and evaluate all dietary classification models:

```bash
python train_dietary_classifier.py --epochs 3 --batch_size 16 --sample_size 1000
```

Parameters:
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--sample_size`: Number of samples to use from dataset (default: 1000)
- `--eval_only`: Skip training and only run evaluation
- `--gpt_samples`: Number of samples to evaluate with GPT-4o-mini (default: 50)

### Using the LabelSense.AI System

Run the main application to analyze food labels:

```bash
python LabelSense.py
```

This will:
1. Load the dietary classification model if available
2. Allow you to analyze a sample image or upload your own
3. Extract ingredients using OCR
4. Classify the ingredients as vegan, vegetarian, or non-vegetarian
5. Identify potential allergens

## Results

After running the training and evaluation script, you'll find:

1. A PDF report with results tables and analysis in `results/dietary_classification_analysis.pdf`
2. Raw evaluation results in `results/dietary_classification_results.json`
3. Visualization images:
   - `training_stats.png`: Learning curves for the RoBERTa model
   - `confusion_matrices.png`: Confusion matrices for each category
   - `model_comparison.png`: Comparison of metrics across all models
   - `per_class_*.png`: Per-class precision, recall, and F1 scores

## Evaluation Metrics

The following metrics are reported for each model:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score for multi-label classification
- **Hamming Loss**: Ratio of incorrect labels to total labels
- **ROC AUC**: Area under the ROC curve for each class

## Analysis

The PDF report provides detailed analysis of:

1. Comparative performance of the three approaches
2. Class-wise performance metrics
3. Trade-offs between model complexity and performance
4. Areas for improvement and future work

## Integration with Main System

The dietary classification models are automatically integrated with the main LabelSense.AI system. When analyzing a food label, the system will:

1. Extract ingredients using OCR
2. Use the trained model to classify dietary suitability
3. Provide results indicating if the product is vegan, vegetarian, or non-vegetarian
4. Show which model was used for the classification (transformer-based, rule-based, or fallback)

## Limitations and Future Work

1. The current model is trained on synthetic data; performance would improve with real-world labeled data.
2. Context understanding could be improved to handle complex ingredient descriptions.
3. Integration with ingredient databases would provide more accurate classifications.
4. User feedback could be incorporated to improve model performance over time. 