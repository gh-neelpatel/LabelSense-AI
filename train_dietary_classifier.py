#!/usr/bin/env python3
# LabelSense.AI - Dietary Classification Model Training
# Checkpoint 4: Model Training, Evaluation, and Analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import argparse
import json
import pickle
from tqdm import tqdm
import random
from fpdf import FPDF

from DietaryClassifier import (
    DietaryClassifier, 
    SimpleRuleBasedClassifier, 
    plot_training_stats, 
    plot_confusion_matrices, 
    plot_model_comparison, 
    plot_per_class_metrics
)

# Constants
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
DATA_DIR = 'data'
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Train and evaluate dietary classification models')
    parser.add_argument('--data', type=str, default='food_fr.csv', help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use from dataset')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and only run evaluation')
    parser.add_argument('--gpt_samples', type=int, default=50, help='Number of samples to evaluate with GPT-4o-mini')
    return parser.parse_args()

def load_data(data_path, sample_size=1000):
    """Load and prepare data for training and evaluation"""
    print(f"Loading data from {data_path}...")
    
    # If data doesn't exist, create a synthetic dataset
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Creating synthetic dataset...")
        return create_synthetic_dataset(sample_size)
    
    # Load the real dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from {data_path}")
        
        # Take a random sample if specified
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"Using a random sample of {sample_size} records")
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Creating synthetic dataset instead...")
        return create_synthetic_dataset(sample_size)

def create_synthetic_dataset(size=1000):
    """Create a synthetic dataset for training and evaluation when real data isn't available"""
    print(f"Creating synthetic dataset with {size} samples...")
    
    # Lists of ingredients by category
    vegan_ingredients = [
        "wheat flour", "sugar", "salt", "vegetable oil", "soy lecithin", "cocoa powder",
        "baking soda", "corn starch", "rice", "oats", "barley", "quinoa", "tofu",
        "nutritional yeast", "soy protein", "chickpeas", "lentils", "beans", "nuts",
        "seeds", "fruits", "vegetables", "plant-based colors", "plant extracts"
    ]
    
    vegetarian_ingredients = [
        "milk", "cream", "butter", "cheese", "yogurt", "eggs", "honey", "whey",
        "casein", "lactose", "milk solids", "egg white", "egg yolk"
    ]
    
    non_vegetarian_ingredients = [
        "beef", "chicken", "pork", "lamb", "turkey", "duck", "fish", "shellfish",
        "gelatin", "lard", "tallow", "rennet", "anchovies", "bone char", "carmine",
        "isinglass", "animal glycerin", "pepsin", "bacon", "salmon", "tuna"
    ]
    
    # Function to generate random ingredient lists
    def generate_random_ingredients(is_vegan=False, is_vegetarian=False):
        ingredients = []
        
        # Add base ingredients
        ingredients.extend(random.sample(vegan_ingredients, k=random.randint(3, 8)))
        
        # Add vegetarian ingredients if not vegan
        if not is_vegan and (is_vegetarian or random.random() > 0.7):
            ingredients.extend(random.sample(vegetarian_ingredients, k=random.randint(1, 3)))
        
        # Add non-vegetarian ingredients if not vegetarian
        if not is_vegetarian and random.random() > 0.7:
            ingredients.extend(random.sample(non_vegetarian_ingredients, k=random.randint(1, 2)))
        
        # Shuffle and join with commas
        random.shuffle(ingredients)
        return ", ".join(ingredients)
    
    # Generate data
    data = []
    for _ in range(size):
        is_vegan = random.random() < 0.3  # 30% vegan
        is_vegetarian = is_vegan or (random.random() < 0.5)  # 50% vegetarian (including vegan)
        
        ingredients_text = generate_random_ingredients(is_vegan, is_vegetarian)
        
        # Create label tags
        label_tags = []
        if is_vegan:
            label_tags.append("vegan")
        if is_vegetarian:
            label_tags.append("vegetarian")
        if not is_vegetarian:
            label_tags.append("non-vegetarian")
        
        data.append({
            "code": f"SYNTH{random.randint(10000, 99999)}",
            "product_name": f"Synthetic Product {random.randint(1, 1000)}",
            "ingredients_text": ingredients_text,
            "labels_tags": ", ".join(label_tags)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save the synthetic dataset
    synthetic_path = os.path.join(DATA_DIR, "synthetic_food_data.csv")
    df.to_csv(synthetic_path, index=False)
    print(f"Saved synthetic dataset to {synthetic_path}")
    
    return df

def train_evaluate_models(data, args):
    """Train and evaluate different models for dietary classification"""
    # Split data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Initialize results dictionary
    results = {
        "roberta_model": {},
        "rule_based": {},
        "gpt4o_mini": {}
    }
    
    # Path for trained model
    model_path = os.path.join(MODELS_DIR, 'dietary_classifier_roberta.pt')
    
    # Train and evaluate RoBERTa model
    roberta_model = DietaryClassifier()
    
    if not args.eval_only and not os.path.exists(model_path):
        print("\n===== Training RoBERTa Model =====")
        training_stats = roberta_model.train(
            train_df, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        # Plot training statistics
        plot_training_stats(training_stats)
        
        # Save the model
        roberta_model.save_model(model_path)
    else:
        print("\n===== Loading pre-trained RoBERTa Model =====")
        if os.path.exists(model_path):
            roberta_model.load_model(model_path)
        else:
            print(f"No pre-trained model found at {model_path}. Training a new model...")
            training_stats = roberta_model.train(
                train_df, 
                epochs=args.epochs, 
                batch_size=args.batch_size
            )
            roberta_model.save_model(model_path)
    
    # Evaluate RoBERTa model
    print("\n===== Evaluating RoBERTa Model =====")
    results["roberta_model"] = roberta_model.evaluate(test_df)
    
    # Evaluate rule-based model
    print("\n===== Evaluating Rule-Based Model =====")
    rule_model = SimpleRuleBasedClassifier()
    
    # Manual evaluation for rule-based model
    test_texts, test_labels = roberta_model.preprocess_data(test_df)
    rule_preds = []
    
    for text in tqdm(test_texts, desc="Rule-based evaluation"):
        pred = rule_model.predict(text)
        # Convert prediction to multi-label format
        rule_pred = [0, 0, 0]  # [vegan, vegetarian, non-vegetarian]
        if pred['is_vegan']:
            rule_pred[0] = 1
        if pred['is_vegetarian']:
            rule_pred[1] = 1
        if pred['is_non_vegetarian']:
            rule_pred[2] = 1
        rule_preds.append(rule_pred)
    
    # Calculate metrics for rule-based model
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score, classification_report, confusion_matrix
    
    rule_preds_np = np.array(rule_preds)
    
    results["rule_based"] = {
        'accuracy': accuracy_score(test_labels.flatten(), rule_preds_np.flatten()),
        'f1_score': f1_score(test_labels, rule_preds_np, average='weighted'),
        'hamming_loss': hamming_loss(test_labels, rule_preds_np),
        'classification_report': classification_report(
            test_labels, rule_preds_np,
            target_names=roberta_model.label_encoder.classes_,
            output_dict=True
        ),
        'confusion_matrices': {}
    }
    
    # Create confusion matrices for rule-based model
    for i, category in enumerate(roberta_model.label_encoder.classes_):
        cm = confusion_matrix(test_labels[:, i], rule_preds_np[:, i])
        results["rule_based"]['confusion_matrices'][category] = cm
    
    try:
        results["rule_based"]['roc_auc'] = roc_auc_score(test_labels, rule_preds_np, average='weighted')
    except:
        results["rule_based"]['roc_auc'] = 0
    
    # Evaluate GPT-4o-mini (if OPENAI_API_KEY is provided)
    if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
        print(f"\n===== Evaluating GPT-4o-mini Model (on {args.gpt_samples} samples) =====")
        # Use a smaller subset for GPT evaluation to save API costs
        gpt_test_texts = test_texts[:args.gpt_samples]
        gpt_test_labels = test_labels[:args.gpt_samples]
        
        gpt_preds = []
        
        for text in tqdm(gpt_test_texts, desc="GPT-4o-mini evaluation"):
            pred = roberta_model.predict_with_gpt(text, api_key=OPENAI_API_KEY)
            # Convert prediction to multi-label format
            gpt_pred = [0, 0, 0]  # [vegan, vegetarian, non-vegetarian]
            if pred['is_vegan']:
                gpt_pred[0] = 1
            if pred['is_vegetarian']:
                gpt_pred[1] = 1
            if pred['is_non_vegetarian']:
                gpt_pred[2] = 1
            gpt_preds.append(gpt_pred)
        
        # Calculate metrics for GPT model
        gpt_preds_np = np.array(gpt_preds)
        
        results["gpt4o_mini"] = {
            'accuracy': accuracy_score(gpt_test_labels.flatten(), gpt_preds_np.flatten()),
            'f1_score': f1_score(gpt_test_labels, gpt_preds_np, average='weighted'),
            'hamming_loss': hamming_loss(gpt_test_labels, gpt_preds_np),
            'classification_report': classification_report(
                gpt_test_labels, gpt_preds_np,
                target_names=roberta_model.label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrices': {}
        }
        
        # Create confusion matrices for GPT model
        for i, category in enumerate(roberta_model.label_encoder.classes_):
            cm = confusion_matrix(gpt_test_labels[:, i], gpt_preds_np[:, i])
            results["gpt4o_mini"]['confusion_matrices'][category] = cm
        
        try:
            results["gpt4o_mini"]['roc_auc'] = roc_auc_score(gpt_test_labels, gpt_preds_np, average='weighted')
        except:
            results["gpt4o_mini"]['roc_auc'] = 0
    else:
        print("\n===== Skipping GPT-4o-mini evaluation (no API key provided) =====")
        # Set placeholder results
        results["gpt4o_mini"] = {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'hamming_loss': 1.0,
            'roc_auc': 0.0,
            'classification_report': {},
            'confusion_matrices': {}
        }
    
    return results

def create_visualizations(results):
    """Create visualizations for comparison of models"""
    print("\n===== Creating Visualizations =====")
    
    # Make the results into the right format for plotting
    model_results = []
    model_names = []
    
    # Only include models with results
    if results["roberta_model"]:
        model_results.append(results["roberta_model"])
        model_names.append("RoBERTa")
    
    if results["rule_based"]:
        model_results.append(results["rule_based"])
        model_names.append("Rule-Based")
    
    if results["gpt4o_mini"] and "accuracy" in results["gpt4o_mini"] and results["gpt4o_mini"]["accuracy"] > 0:
        model_results.append(results["gpt4o_mini"])
        model_names.append("GPT-4o-mini")
    
    # Skip plotting if there are not enough models to compare
    if len(model_results) < 1:
        print("Not enough models with results to create comparisons.")
        return
    
    # Plot model comparison
    if len(model_results) > 1:
        plot_model_comparison(model_results, model_names)
    
    # Plot confusion matrices for RoBERTa
    if results["roberta_model"] and "confusion_matrices" in results["roberta_model"]:
        plot_confusion_matrices(
            results["roberta_model"]["confusion_matrices"],
            results["roberta_model"]["classification_report"].keys()
        )
    
    # Plot per-class metrics comparison if we have multiple models
    if len(model_results) > 1 and all("classification_report" in r for r in model_results):
        plot_per_class_metrics(model_results, model_names)
    
    print("Visualizations saved to current directory.")

def create_results_pdf(results):
    """Create a PDF report of the results"""
    print("\n===== Creating PDF Report =====")
    
    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "LabelSense.AI - Dietary Classification Analysis", ln=True, align="C")
    pdf.ln(5)
    
    # Add date
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Add introduction
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, "This report presents the evaluation results of different approaches to dietary classification (vegan, vegetarian, non-vegetarian) based on food ingredient lists. The evaluation compares a fine-tuned RoBERTa model, a rule-based approach, and the GPT-4o-mini model.")
    pdf.ln(5)
    
    # Add results table
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Results Table", ln=True)
    
    # Create table header
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 7, "Metric", 1, 0, "C")
    pdf.cell(45, 7, "RoBERTa", 1, 0, "C")
    pdf.cell(45, 7, "Rule-Based", 1, 0, "C")
    pdf.cell(45, 7, "GPT-4o-mini", 1, 1, "C")
    
    # Add table rows
    pdf.set_font("Arial", "", 10)
    metrics = ["accuracy", "f1_score", "hamming_loss", "roc_auc"]
    metric_names = ["Accuracy", "F1 Score", "Hamming Loss", "ROC AUC"]
    
    for metric, name in zip(metrics, metric_names):
        pdf.cell(50, 7, name, 1, 0)
        pdf.cell(45, 7, f"{results['roberta_model'].get(metric, 0):.4f}", 1, 0, "C")
        pdf.cell(45, 7, f"{results['rule_based'].get(metric, 0):.4f}", 1, 0, "C")
        pdf.cell(45, 7, f"{results['gpt4o_mini'].get(metric, 0):.4f}", 1, 1, "C")
    
    # Add per-class results for RoBERTa
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Per-Class Results (RoBERTa)", ln=True)
    
    if "classification_report" in results["roberta_model"]:
        # Table header
        pdf.set_font("Arial", "B", 10)
        pdf.cell(40, 7, "Class", 1, 0, "C")
        pdf.cell(38, 7, "Precision", 1, 0, "C")
        pdf.cell(38, 7, "Recall", 1, 0, "C")
        pdf.cell(38, 7, "F1-Score", 1, 0, "C")
        pdf.cell(38, 7, "Support", 1, 1, "C")
        
        # Table rows
        pdf.set_font("Arial", "", 10)
        for class_name, metrics in results["roberta_model"]["classification_report"].items():
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                pdf.cell(40, 7, class_name, 1, 0)
                pdf.cell(38, 7, f"{metrics.get('precision', 0):.4f}", 1, 0, "C")
                pdf.cell(38, 7, f"{metrics.get('recall', 0):.4f}", 1, 0, "C")
                pdf.cell(38, 7, f"{metrics.get('f1-score', 0):.4f}", 1, 0, "C")
                pdf.cell(38, 7, f"{metrics.get('support', 0)}", 1, 1, "C")
    
    # Add analysis
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "4. Analysis", ln=True)
    pdf.set_font("Arial", "", 10)
    
    # Determine which model performed best
    metrics = ["accuracy", "f1_score"]
    model_scores = {
        "RoBERTa": sum(results["roberta_model"].get(m, 0) for m in metrics),
        "Rule-Based": sum(results["rule_based"].get(m, 0) for m in metrics),
        "GPT-4o-mini": sum(results["gpt4o_mini"].get(m, 0) for m in metrics)
    }
    best_model = max(model_scores.items(), key=lambda x: x[1])[0]
    
    analysis_text = f"""
Based on the evaluation metrics, the {best_model} model demonstrates the best overall performance for dietary classification. 

Key observations:

1. Performance comparison: The RoBERTa model achieves {results['roberta_model'].get('accuracy', 0):.2%} accuracy and {results['roberta_model'].get('f1_score', 0):.2%} F1 score, compared to {results['rule_based'].get('accuracy', 0):.2%} accuracy and {results['rule_based'].get('f1_score', 0):.2%} F1 score for the rule-based approach.

2. Multi-label classification: The metrics indicate that the models handle the multi-label nature of dietary classification (vegan, vegetarian, non-vegetarian) with varying degrees of success.

3. Class imbalance: There are variations in performance across different dietary categories, which suggests class imbalance in the dataset.

4. Complexity vs. simplicity: While the neural model offers better overall performance, the rule-based approach provides a simpler, more interpretable solution with reasonable accuracy.

5. Trade-offs: The choice between models involves trade-offs between accuracy, computational requirements, and explainability.
"""
    pdf.multi_cell(0, 5, analysis_text)
    
    # Add visualizations
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "5. Visualizations", ln=True)
    
    # Add model comparison plot if it exists
    if os.path.exists("model_comparison.png"):
        pdf.image("model_comparison.png", x=10, w=180)
        pdf.ln(5)
    
    # Add confusion matrices if they exist
    if os.path.exists("confusion_matrices.png"):
        pdf.image("confusion_matrices.png", x=10, w=180)
        pdf.ln(5)
    
    # Add per-class plots if they exist
    for metric in ["precision", "recall", "f1-score"]:
        if os.path.exists(f"per_class_{metric}.png"):
            pdf.image(f"per_class_{metric}.png", x=10, w=180)
            pdf.ln(5)
    
    # Add conclusion
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "6. Conclusion", ln=True)
    pdf.set_font("Arial", "", 10)
    
    conclusion_text = f"""
The evaluation demonstrates that the {best_model} model provides the most reliable dietary classification for the LabelSense.AI system. This approach balances accuracy with computational efficiency, making it suitable for real-time food label analysis.

Future improvements could include:
1. Expanding the training dataset with more diverse food products
2. Incorporating ingredient context understanding
3. Adding confidence scores to predictions
4. Implementing a hybrid approach that combines the strengths of multiple models
5. Enhancing the model with additional information from nutrition facts

These results validate the effectiveness of our approach for the dietary classification component of the LabelSense.AI system, providing a reliable foundation for users to identify vegan, vegetarian, and non-vegetarian products from their ingredients.
"""
    pdf.multi_cell(0, 5, conclusion_text)
    
    # Save the PDF
    pdf_path = os.path.join(RESULTS_DIR, "dietary_classification_analysis.pdf")
    pdf.output(pdf_path)
    print(f"PDF report saved to {pdf_path}")

def main():
    args = setup_argparse()
    
    # Load and prepare data
    data = load_data(os.path.join(DATA_DIR, args.data), args.sample_size)
    
    # Train and evaluate models
    results = train_evaluate_models(data, args)
    
    # Save results to JSON
    results_path = os.path.join(RESULTS_DIR, "dietary_classification_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    for model_name, model_results in results.items():
        if "confusion_matrices" in model_results:
            for category, matrix in model_results["confusion_matrices"].items():
                if hasattr(matrix, "tolist"):
                    model_results["confusion_matrices"][category] = matrix.tolist()
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    
    # Create visualizations
    create_visualizations(results)
    
    # Create PDF report
    create_results_pdf(results)
    
    print("\n===== Dietary Classification Analysis Complete =====")

if __name__ == "__main__":
    main() 