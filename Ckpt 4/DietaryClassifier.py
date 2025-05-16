import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, hamming_loss, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from PIL import Image
import cv2
import pickle
import json
import re
import time
import random
import openai
from tqdm import tqdm

# Seed everything for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
DIETARY_CATEGORIES = ['vegan', 'vegetarian', 'non-vegetarian']

class IngredientDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class DietaryClassifier:
    def __init__(self, model_path=None):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = MultiLabelBinarizer()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self):
        """Initialize the RoBERTa model for dietary classification"""
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = len(DIETARY_CATEGORIES)
        
        model = RobertaModel.from_pretrained('roberta-base', config=config)
        
        # Add classification head
        class DietaryClassificationModel(torch.nn.Module):
            def __init__(self, roberta_model, num_labels):
                super(DietaryClassificationModel, self).__init__()
                self.roberta = roberta_model
                self.dropout = torch.nn.Dropout(0.1)
                self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, num_labels)
                self.sigmoid = torch.nn.Sigmoid()
                
            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                sequence_output = outputs[0]
                pooled_output = sequence_output[:, 0, :]  # [CLS] token
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return self.sigmoid(logits)
        
        self.model = DietaryClassificationModel(model, len(DIETARY_CATEGORIES))
        self.model.to(device)
        return self.model
    
    def preprocess_data(self, data, ingredients_col='ingredients_text', labels_col='labels_tags'):
        """Preprocess the data for training"""
        # Clean ingredients text
        ingredients = data[ingredients_col].fillna('').apply(lambda x: re.sub(r'\s+', ' ', x.lower().strip()))
        
        # Process dietary labels
        labels = []
        for label_text in data[labels_col].fillna(''):
            label_set = []
            # Check for vegan
            if 'vegan' in label_text.lower():
                label_set.append('vegan')
                label_set.append('vegetarian')  # Vegan is also vegetarian
            # Check for vegetarian
            elif 'vegetarian' in label_text.lower():
                label_set.append('vegetarian')
            # If neither, assume non-vegetarian
            else:
                label_set.append('non-vegetarian')
            labels.append(label_set)
        
        # Binarize the labels
        self.label_encoder.fit([['vegan'], ['vegetarian'], ['non-vegetarian']])
        binary_labels = self.label_encoder.transform(labels)
        
        return ingredients.tolist(), binary_labels
    
    def train(self, train_data, val_data=None, epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the dietary classification model"""
        print("Preprocessing training data...")
        
        # Preprocess training data
        if val_data is None:
            train_texts, train_labels = self.preprocess_data(train_data)
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.1, random_state=42
            )
        else:
            train_texts, train_labels = self.preprocess_data(train_data)
            val_texts, val_labels = self.preprocess_data(val_data)
        
        # Create datasets
        train_dataset = IngredientDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = IngredientDataset(val_texts, val_labels, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_stats = []
        
        print(f"Starting training on {device}...")
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Binary cross entropy loss
                loss = torch.nn.BCELoss()(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []
            
            progress_bar = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Binary cross entropy loss
                    loss = torch.nn.BCELoss()(outputs, labels)
                    
                    total_val_loss += loss.item()
                    
                    # Convert predictions to binary (0 or 1)
                    preds = (outputs > 0.5).float().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Calculate metrics
            accuracy = accuracy_score(np.array(all_labels).flatten(), np.array(all_preds).flatten())
            f1 = f1_score(np.array(all_labels), np.array(all_preds), average='weighted')
            
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1 Score: {f1:.4f}")
            
            # Save stats
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': accuracy,
                'val_f1': f1
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model('best_dietary_model.pt')
        
        return training_stats
    
    def predict(self, ingredients_text):
        """Predict dietary classification for ingredients text"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Clean and prepare text
        if isinstance(ingredients_text, list):
            texts = [re.sub(r'\s+', ' ', text.lower().strip()) for text in ingredients_text]
        else:
            texts = [re.sub(r'\s+', ' ', ingredients_text.lower().strip())]
        
        # Create dataset
        dataset = IngredientDataset(texts, np.zeros((len(texts), len(DIETARY_CATEGORIES))), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Make predictions
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Convert predictions to binary (0 or 1)
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
        
        # Convert predictions to labels
        pred_labels = self.label_encoder.inverse_transform(np.array(all_preds))
        
        # Return results for each input
        results = []
        for idx, pred in enumerate(pred_labels):
            result = {
                'is_vegan': 'vegan' in pred,
                'is_vegetarian': 'vegetarian' in pred,
                'is_non_vegetarian': 'non-vegetarian' in pred,
                'raw_prediction': pred
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        print("Evaluating dietary classifier...")
        
        # Preprocess test data
        test_texts, test_labels = self.preprocess_data(test_data)
        
        # Create dataset
        test_dataset = IngredientDataset(test_texts, test_labels, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        # Model evaluation
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Convert predictions to binary (0 or 1)
                preds = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels_np.flatten(), all_preds_np.flatten())
        f1 = f1_score(all_labels_np, all_preds_np, average='weighted')
        hamming = hamming_loss(all_labels_np, all_preds_np)
        
        try:
            roc_auc = roc_auc_score(all_labels_np, all_preds_np, average='weighted')
        except:
            roc_auc = 0  # For cases where there's only one class
        
        # Generate classification report for each category
        class_names = self.label_encoder.classes_
        report = classification_report(
            all_labels_np, all_preds_np, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Create confusion matrices
        conf_matrices = {}
        for i, category in enumerate(class_names):
            cm = confusion_matrix(all_labels_np[:, i], all_preds_np[:, i])
            conf_matrices[category] = cm
        
        # Combine results
        eval_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'hamming_loss': hamming,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrices': conf_matrices
        }
        
        return eval_results
    
    def save_model(self, filepath):
        """Save the trained model and tokenizer"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and tokenizer"""
        if not os.path.exists(filepath):
            raise ValueError(f"Model file {filepath} not found")
        
        # Load state dict
        checkpoint = torch.load(filepath, map_location=device)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"Model loaded from {filepath}")

    def predict_with_gpt(self, ingredients_text, api_key=None):
        """Use GPT-4o-mini for ingredient classification"""
        # Set API key
        if api_key:
            openai.api_key = api_key
        
        # Prepare the message
        prompt = f"""
        Analyze the following food ingredients and determine if the food is:
        1. Vegan (contains no animal products)
        2. Vegetarian (may contain dairy/eggs but no meat/fish)
        3. Non-vegetarian (contains meat, fish, or other animal products)
        
        Ingredients: {ingredients_text}
        
        Respond in JSON format with keys "is_vegan", "is_vegetarian", and "is_non_vegetarian" (all boolean values).
        Also include "explanation" with a brief reason for the classification.
        """
        
        try:
            # Call the GPT API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes food ingredients to determine their dietary classification."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON
            try:
                # Find JSON in the text if wrapped in other text
                pattern = r'\{.*\}'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    content = match.group(0)
                
                result = json.loads(content)
                
                # Ensure required keys are present
                required_keys = ["is_vegan", "is_vegetarian", "is_non_vegetarian", "explanation"]
                for key in required_keys:
                    if key not in result:
                        result[key] = key == "explanation" and "No explanation provided" or False
                
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                return {
                    "is_vegan": "vegan" in content.lower(),
                    "is_vegetarian": "vegetarian" in content.lower() and "non-vegetarian" not in content.lower(),
                    "is_non_vegetarian": "non-vegetarian" in content.lower(),
                    "explanation": content.strip(),
                    "raw_response": content
                }
        except Exception as e:
            return {
                "is_vegan": False,
                "is_vegetarian": False, 
                "is_non_vegetarian": True,
                "explanation": f"Error calling GPT API: {str(e)}",
                "error": str(e)
            }

class SimpleRuleBasedClassifier:
    def __init__(self):
        self.non_vegan = [
            'milk', 'dairy', 'cheese', 'eggs', 'egg', 'honey', 'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 
            'gelatin', 'gelatine', 'lard', 'tallow', 'whey', 'casein', 'lactose', 'rennet', 'shellac', 'carmine', 
            'isinglass', 'albumin', 'cochineal', 'fish', 'shellfish', 'beef fat', 'butter', 'buttermilk', 'yogurt', 
            'cream', 'mayonnaise', 'bacon', 'duck', 'goose'
        ]
        
        self.non_vegetarian = [
            'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 'gelatin', 'gelatine', 'lard', 'tallow', 'rennet',
            'shellac', 'isinglass', 'carmine', 'cochineal', 'fish', 'shellfish', 'beef fat', 'bacon', 'duck', 'goose'
        ]
    
    def predict(self, ingredients_text):
        """Simple rule-based prediction"""
        if isinstance(ingredients_text, list):
            return [self._predict_single(text) for text in ingredients_text]
        return self._predict_single(ingredients_text)
    
    def _predict_single(self, ingredients_text):
        """Predict for a single text input"""
        ingredients_text = ingredients_text.lower()
        
        # Check if contains non-vegetarian ingredients
        is_non_vegetarian = any(item in ingredients_text for item in self.non_vegetarian)
        
        # Check if contains non-vegan ingredients
        is_non_vegan = any(item in ingredients_text for item in self.non_vegan)
        
        # Determine classification
        is_vegetarian = not is_non_vegetarian
        is_vegan = not is_non_vegan and is_vegetarian
        
        return {
            'is_vegan': is_vegan,
            'is_vegetarian': is_vegetarian,
            'is_non_vegetarian': is_non_vegetarian
        }

# Utility functions to visualize results
def plot_training_stats(stats):
    """Plot training statistics"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([stat['epoch'] for stat in stats], [stat['train_loss'] for stat in stats], 'b-o', label='Training Loss')
    plt.plot([stat['epoch'] for stat in stats], [stat['val_loss'] for stat in stats], 'r-o', label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot([stat['epoch'] for stat in stats], [stat['val_accuracy'] for stat in stats], 'g-o', label='Accuracy')
    plt.plot([stat['epoch'] for stat in stats], [stat['val_f1'] for stat in stats], 'p-o', label='F1 Score')
    plt.title('Metrics over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_stats.png')
    plt.close()

def plot_confusion_matrices(conf_matrices, class_names):
    """Plot confusion matrices for each category"""
    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(15, 5))
    
    for i, (category, cm) in enumerate(conf_matrices.items()):
        ax = axes[i] if len(conf_matrices) > 1 else axes
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_title(f'Confusion Matrix: {category}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def plot_model_comparison(results, model_names):
    """Plot comparison of different models"""
    metrics = ['accuracy', 'f1_score', 'hamming_loss', 'roc_auc']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [result[metric] for result in results]
        axes[i].bar(model_names, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylim(0, 1)
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_per_class_metrics(results, model_names):
    """Plot per-class metrics for different models"""
    categories = list(results[0]['classification_report'].keys())
    categories = [c for c in categories if c not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for i, category in enumerate(categories):
            values = []
            for result in results:
                values.append(result['classification_report'][category][metric])
            
            x = np.arange(len(model_names))
            width = 0.8 / len(categories)
            offset = (i - len(categories)/2 + 0.5) * width
            
            plt.bar(x + offset, values, width, label=category)
        
        plt.xlabel('Models')
        plt.ylabel(f'{metric}')
        plt.title(f'Per-Class {metric.title()} Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'per_class_{metric}.png')
        plt.close() 