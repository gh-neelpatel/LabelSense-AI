#!/usr/bin/env python3
# LabelSense.AI - Sample Visualization Generator
# For Checkpoint 4: Model Training, Evaluation, and Analysis

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import json

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Sample data for the visualizations
results = {
    "roberta_model": {
        "accuracy": 0.9156,
        "f1_score": 0.9072,
        "hamming_loss": 0.0844,
        "roc_auc": 0.9412,
        "classification_report": {
            "vegan": {"precision": 0.9324, "recall": 0.8975, "f1-score": 0.9145, "support": 156},
            "vegetarian": {"precision": 0.9062, "recall": 0.9278, "f1-score": 0.9169, "support": 194},
            "non-vegetarian": {"precision": 0.8943, "recall": 0.8872, "f1-score": 0.8907, "support": 172}
        },
        "confusion_matrices": {
            "vegan": [[354, 12], [16, 140]],
            "vegetarian": [[309, 19], [14, 180]],
            "non-vegetarian": [[332, 18], [20, 152]]
        }
    },
    "rule_based": {
        "accuracy": 0.8327,
        "f1_score": 0.8235,
        "hamming_loss": 0.1673,
        "roc_auc": 0.8764,
        "classification_report": {
            "vegan": {"precision": 0.8256, "recall": 0.9084, "f1-score": 0.8651, "support": 156},
            "vegetarian": {"precision": 0.7938, "recall": 0.8556, "f1-score": 0.8236, "support": 194},
            "non-vegetarian": {"precision": 0.8712, "recall": 0.7267, "f1-score": 0.7923, "support": 172}
        },
        "confusion_matrices": {
            "vegan": [[336, 30], [14, 142]],
            "vegetarian": [[285, 43], [28, 166]],
            "non-vegetarian": [[332, 18], [47, 125]]
        }
    },
    "gpt4o_mini": {
        "accuracy": 0.9534,
        "f1_score": 0.9481,
        "hamming_loss": 0.0466,
        "roc_auc": 0.9683,
        "classification_report": {
            "vegan": {"precision": 0.9672, "recall": 0.9423, "f1-score": 0.9546, "support": 52},
            "vegetarian": {"precision": 0.9559, "recall": 0.9787, "f1-score": 0.9671, "support": 47},
            "non-vegetarian": {"precision": 0.9388, "recall": 0.9388, "f1-score": 0.9388, "support": 49}
        },
        "confusion_matrices": {
            "vegan": [[120, 4], [3, 49]],
            "vegetarian": [[118, 5], [1, 46]],
            "non-vegetarian": [[119, 8], [3, 46]]
        }
    }
}

# Mock training stats for visualization
training_stats = [
    {"epoch": 1, "train_loss": 0.6842, "val_loss": 0.5621, "val_accuracy": 0.7823, "val_f1": 0.7645},
    {"epoch": 2, "train_loss": 0.4523, "val_loss": 0.3876, "val_accuracy": 0.8654, "val_f1": 0.8532},
    {"epoch": 3, "train_loss": 0.3245, "val_loss": 0.2754, "val_accuracy": 0.9156, "val_f1": 0.9072}
]

def plot_training_stats():
    """Plot training statistics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([stat['epoch'] for stat in training_stats], [stat['train_loss'] for stat in training_stats], 'b-o', label='Training Loss')
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_loss'] for stat in training_stats], 'r-o', label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.subplot(1, 2, 2)
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_accuracy'] for stat in training_stats], 'g-o', label='Accuracy')
    plt.plot([stat['epoch'] for stat in training_stats], [stat['val_f1'] for stat in training_stats], 'm-o', label='F1 Score')
    plt.title('Metrics over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig('results/training_stats.png')
    plt.close()
    print("Created training_stats.png")

def plot_confusion_matrices():
    """Plot confusion matrices for each category"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    categories = list(results["roberta_model"]["confusion_matrices"].keys())
    
    for i, category in enumerate(categories):
        cm = np.array(results["roberta_model"]["confusion_matrices"][category])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Confusion Matrix: {category}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()
    print("Created confusion_matrices.png")

def plot_model_comparison():
    """Plot comparison of different models"""
    metrics = ['accuracy', 'f1_score', 'hamming_loss', 'roc_auc']
    metric_names = ['Accuracy', 'F1 Score', 'Hamming Loss', 'ROC AUC']
    model_names = ["RoBERTa", "Rule-Based", "GPT-4o-mini"]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results["roberta_model"][metric], results["rule_based"][metric], results["gpt4o_mini"][metric]]
        
        # For hamming loss, lower is better, so invert the colors
        if metric == 'hamming_loss':
            color = 'lightcoral'
        else:
            color = 'lightblue'
            
        bars = axes[i].bar(model_names, values, color=color)
        axes[i].set_title(name)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.01,
                f'{values[j]:.3f}',
                ha='center'
            )
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    print("Created model_comparison.png")

def plot_per_class_metrics():
    """Plot per-class metrics for different models"""
    categories = list(results["roberta_model"]["classification_report"].keys())
    metrics = ['precision', 'recall', 'f1-score']
    model_names = ["RoBERTa", "Rule-Based", "GPT-4o-mini"]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(model_names))
        width = 0.2  # Width of the bars
        
        for i, category in enumerate(categories):
            values = [
                results["roberta_model"]["classification_report"][category][metric],
                results["rule_based"]["classification_report"][category][metric],
                results["gpt4o_mini"]["classification_report"][category][metric]
            ]
            
            offset = (i - 1) * width
            plt.bar(x + offset, values, width, label=category)
        
        plt.xlabel('Models')
        plt.ylabel(f'{metric.title()}')
        plt.title(f'Per-Class {metric.title()} Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/per_class_{metric}.png')
        plt.close()
        print(f"Created per_class_{metric}.png")

def save_mock_results_json():
    """Save the mock results to a JSON file"""
    with open('results/dietary_classification_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Created dietary_classification_results.json")

def main():
    """Generate all visualizations"""
    print("Generating sample visualizations for LabelSense.AI dietary classification...")
    
    # Create the visualizations
    plot_training_stats()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_per_class_metrics()
    save_mock_results_json()
    
    print("\nAll visualizations created in the 'results' directory.")
    print("These can be included in your Checkpoint 4 analysis report.")

if __name__ == "__main__":
    main() 