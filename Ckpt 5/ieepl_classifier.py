#!/usr/bin/env python3
"""
LabelSense.AI - Checkpoint 5 Innovation Implementation
Intelligent Ensemble with Error-Pattern Learning (IEEPL)
"""

import pandas as pd
import numpy as np
import json
import re
import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor:
    """Extract features that correlate with model reliability"""
    
    def __init__(self):
        self.cultural_markers = {
            'asian': ['sake', 'miso', 'soy sauce', 'fish sauce', 'dashi', 'kombu'],
            'indian': ['ghee', 'paneer', 'garam masala', 'turmeric', 'cardamom'],
            'middle_eastern': ['tahini', 'sumac', 'za\'atar', 'harissa'],
            'latin': ['queso', 'chorizo', 'chipotle', 'adobo'],
            'european': ['prosciutto', 'pancetta', 'gruyere', 'brie']
        }
        
    def extract_reliability_features(self, ingredients_text):
        """Extract features that predict model reliability"""
        ingredients_text = ingredients_text.lower()
        words = ingredients_text.split()
        
        features = {
            # Basic text statistics
            'ingredient_count': len(ingredients_text.split(',')),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'text_length': len(ingredients_text),
            
            # Pattern indicators (convert to int)
            'has_processing_terms': int(bool(re.search(r'extract|isolate|concentrate|powder|protein', ingredients_text))),
            'has_ambiguous_terms': int(bool(re.search(r'natural flavor|artificial flavor|spices|seasoning', ingredients_text))),
            'has_obvious_animal': int(bool(re.search(r'meat|beef|pork|chicken|fish|dairy|milk|cream|butter|cheese|eggs', ingredients_text))),
            'has_obvious_plant': int(bool(re.search(r'vegetable|fruit|grain|bean|nuts|seeds|oil|vinegar', ingredients_text))),
            
            # Cultural markers (convert to int)
            'has_cultural_markers': int(self._has_cultural_markers(ingredients_text)),
            'cultural_region_code': self._get_cultural_region_code(ingredients_text),
            
            # Complexity indicators
            'parentheses_count': ingredients_text.count('('),
            'comma_count': ingredients_text.count(','),
            'complexity_score': self._calculate_complexity_score(ingredients_text),
            
            # Specific problematic patterns from Checkpoint 4 (convert to int)
            'has_novel_ingredients': int(bool(re.search(r'nutritional yeast|tapioca|pea protein|quinoa|chia|hemp', ingredients_text))),
            'has_extracts': ingredients_text.count('extract'),
            'has_numbers': int(bool(re.search(r'\d', ingredients_text))),
        }
        
        return features
    
    def _has_cultural_markers(self, text):
        """Check if text contains cultural food markers"""
        for region, markers in self.cultural_markers.items():
            if any(marker in text for marker in markers):
                return True
        return False
    
    def _get_cultural_region_code(self, text):
        """Get numeric code for cultural region"""
        region_codes = {'western': 0, 'asian': 1, 'indian': 2, 'middle_eastern': 3, 'latin': 4, 'european': 5}
        
        for region, markers in self.cultural_markers.items():
            if any(marker in text for marker in markers):
                return region_codes[region]
        return region_codes['western']
    
    def _calculate_complexity_score(self, text):
        """Calculate complexity score based on various factors"""
        score = 0
        score += len(text.split(',')) * 0.1  # Number of ingredients
        score += text.count('(') * 0.2  # Parenthetical information
        score += len([w for w in text.split() if len(w) > 10]) * 0.3  # Long words
        score += bool(re.search(r'extract|isolate|concentrate', text)) * 0.5  # Processing terms
        return min(score, 5.0)  # Cap at 5.0

class ReliabilityPredictor:
    """Predict which model is most reliable for given input"""
    
    def __init__(self):
        self.rule_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.roberta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
    def prepare_training_data(self, error_data_path='misclassification_analysis_results.json'):
        """Prepare training data from Checkpoint 4 error analysis"""
        try:
            with open(error_data_path, 'r') as f:
                error_data = json.load(f)
        except FileNotFoundError:
            # Create sample data if file doesn't exist
            print("Error data file not found, creating sample data...")
            return self._create_sample_training_data()
        
        training_examples = []
        
        # Process rule-based errors (where RoBERTa was correct)
        for example in error_data['examples']['rule_errors']:
            features = self.feature_extractor.extract_reliability_features(example['ingredients'])
            training_examples.append({
                'features': features,
                'rule_correct': False,
                'roberta_correct': True,
                'ingredients': example['ingredients']
            })
        
        # Process RoBERTa errors (where rule-based was correct)
        for example in error_data['examples']['roberta_errors']:
            features = self.feature_extractor.extract_reliability_features(example['ingredients'])
            training_examples.append({
                'features': features,
                'rule_correct': True,
                'roberta_correct': False,
                'ingredients': example['ingredients']
            })
        
        # Add some correct cases for both models
        self._add_correct_cases(training_examples)
        
        return training_examples
    
    def _create_sample_training_data(self):
        """Create sample training data for demonstration"""
        sample_data = [
            # Rule-based errors
            {'ingredients': 'pea protein isolate, natural flavors', 'rule_correct': False, 'roberta_correct': True},
            {'ingredients': 'nutritional yeast, cashew cream', 'rule_correct': False, 'roberta_correct': True},
            {'ingredients': 'mushroom extract, sea salt', 'rule_correct': False, 'roberta_correct': True},
            {'ingredients': 'tapioca starch, agar, natural coconut flavor', 'rule_correct': False, 'roberta_correct': True},
            {'ingredients': 'quinoa flour, hemp seeds, stevia extract', 'rule_correct': False, 'roberta_correct': True},
            
            # RoBERTa errors  
            {'ingredients': 'cream, salt', 'rule_correct': True, 'roberta_correct': False},
            {'ingredients': 'honey, oats', 'rule_correct': True, 'roberta_correct': False},
            {'ingredients': 'anchovies, water', 'rule_correct': True, 'roberta_correct': False},
            {'ingredients': 'lard', 'rule_correct': True, 'roberta_correct': False},
            {'ingredients': 'gellan gum', 'rule_correct': True, 'roberta_correct': False},
            
            # Both correct
            {'ingredients': 'wheat flour, sugar, vegetable oil', 'rule_correct': True, 'roberta_correct': True},
            {'ingredients': 'chicken breast, salt, pepper', 'rule_correct': True, 'roberta_correct': True},
            {'ingredients': 'tomatoes, basil, olive oil', 'rule_correct': True, 'roberta_correct': True},
            {'ingredients': 'milk, sugar, vanilla', 'rule_correct': True, 'roberta_correct': True},
        ]
        
        training_examples = []
        for item in sample_data:
            features = self.feature_extractor.extract_reliability_features(item['ingredients'])
            training_examples.append({
                'features': features,
                'rule_correct': item['rule_correct'],
                'roberta_correct': item['roberta_correct'],
                'ingredients': item['ingredients']
            })
        
        return training_examples
    
    def _add_correct_cases(self, training_examples):
        """Add cases where both models are correct for balance"""
        correct_cases = [
            'wheat flour, sugar, eggs, butter',  # Obviously vegetarian
            'beef, salt, pepper',  # Obviously non-vegetarian
            'tofu, soy sauce, vegetables',  # Obviously vegan
            'milk, sugar, vanilla',  # Obviously vegetarian
        ]
        
        for ingredients in correct_cases:
            features = self.feature_extractor.extract_reliability_features(ingredients)
            training_examples.append({
                'features': features,
                'rule_correct': True,
                'roberta_correct': True,
                'ingredients': ingredients
            })
    
    def train(self, training_data):
        """Train reliability prediction models"""
        print("Training reliability predictors...")
        
        # Convert features to numeric arrays
        feature_names = list(training_data[0]['features'].keys())
        X = []
        y_rule = []
        y_roberta = []
        
        for example in training_data:
            # Ensure all feature values are numeric
            feature_vector = []
            for name in feature_names:
                value = example['features'][name]
                # Convert boolean to int if needed
                if isinstance(value, bool):
                    value = int(value)
                feature_vector.append(float(value))
            
            X.append(feature_vector)
            y_rule.append(int(example['rule_correct']))
            y_roberta.append(int(example['roberta_correct']))
        
        X = np.array(X)
        y_rule = np.array(y_rule)
        y_roberta = np.array(y_roberta)
        
        print(f"Training on {len(X)} examples with {len(feature_names)} features")
        
        # Train models
        self.rule_model.fit(X, y_rule)
        self.roberta_model.fit(X, y_roberta)
        self.feature_names = feature_names
        self.is_trained = True
        
        # Print feature importance
        print("\nFeature Importance for Rule-Based Reliability:")
        rule_importances = list(zip(feature_names, self.rule_model.feature_importances_))
        rule_importances.sort(key=lambda x: x[1], reverse=True)
        for name, importance in rule_importances[:5]:
            print(f"  {name}: {importance:.3f}")
        
        print("\nFeature Importance for RoBERTa Reliability:")
        roberta_importances = list(zip(feature_names, self.roberta_model.feature_importances_))
        roberta_importances.sort(key=lambda x: x[1], reverse=True)
        for name, importance in roberta_importances[:5]:
            print(f"  {name}: {importance:.3f}")
    
    def predict_reliability(self, ingredients_text):
        """Predict reliability of each model for given input"""
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        features = self.feature_extractor.extract_reliability_features(ingredients_text)
        
        # Convert to numeric array
        feature_vector = []
        for name in self.feature_names:
            value = features[name]
            if isinstance(value, bool):
                value = int(value)
            feature_vector.append(float(value))
        
        X = np.array([feature_vector])
        
        rule_reliability = self.rule_model.predict_proba(X)[0][1]  # Probability of being correct
        roberta_reliability = self.roberta_model.predict_proba(X)[0][1]
        
        return rule_reliability, roberta_reliability, features

class ConflictResolver:
    """Resolve conflicts between model predictions"""
    
    def __init__(self, conflict_threshold=0.3):
        self.conflict_threshold = conflict_threshold
    
    def resolve_conflict(self, rule_pred, roberta_pred, features, rule_reliability, roberta_reliability):
        """Resolve conflict between model predictions"""
        
        # Check if there's a significant disagreement
        pred_diff = abs(self._get_pred_score(rule_pred) - self._get_pred_score(roberta_pred))
        
        if pred_diff < self.conflict_threshold:
            # No significant conflict, use weighted average
            weights = [rule_reliability, roberta_reliability]
            weights = np.array(weights) / sum(weights)
            return self._weighted_prediction(rule_pred, roberta_pred, weights), "weighted_average"
        
        # Significant conflict detected - use heuristics
        if features['ingredient_count'] < 3:
            # Short lists - trust rule-based
            return rule_pred, "short_list_heuristic"
        elif features['has_cultural_markers']:
            # Cultural foods - trust RoBERTa
            return roberta_pred, "cultural_heuristic"
        elif features['has_ambiguous_terms']:
            # Ambiguous ingredients - flag for review
            return "requires_review", "ambiguity_detected"
        elif rule_reliability > roberta_reliability * 1.5:
            return rule_pred, "rule_confidence"
        elif roberta_reliability > rule_reliability * 1.5:
            return roberta_pred, "roberta_confidence"
        else:
            # Can't resolve - flag for review
            return "requires_review", "unresolvable_conflict"
    
    def _get_pred_score(self, prediction):
        """Convert prediction to numeric score for comparison"""
        if isinstance(prediction, dict):
            if 'vegan' in prediction['dietary_category']:
                return 2
            elif 'vegetarian' in prediction['dietary_category']:
                return 1
            else:
                return 0
        elif isinstance(prediction, str):
            if 'vegan' in prediction.lower():
                return 2
            elif 'vegetarian' in prediction.lower():
                return 1
            else:
                return 0
        return 0
    
    def _weighted_prediction(self, rule_pred, roberta_pred, weights):
        """Combine predictions using weights"""
        rule_score = self._get_pred_score(rule_pred)
        roberta_score = self._get_pred_score(roberta_pred)
        
        weighted_score = rule_score * weights[0] + roberta_score * weights[1]
        
        if weighted_score >= 1.5:
            return "Vegan"
        elif weighted_score >= 0.5:
            return "Vegetarian"  
        else:
            return "Non-Vegetarian"

class IEEPLClassifier:
    """Main Intelligent Ensemble with Error-Pattern Learning classifier"""
    
    def __init__(self):
        self.reliability_predictor = ReliabilityPredictor()
        self.conflict_resolver = ConflictResolver()
        self.is_trained = False
        
        # Mock models for demonstration (in real implementation, load actual models)
        self.rule_model = self._create_mock_rule_model()
        self.roberta_model = self._create_mock_roberta_model()
    
    def _create_mock_rule_model(self):
        """Create mock rule-based model for demonstration"""
        class MockRuleModel:
            def predict(self, ingredients):
                ingredients = ingredients.lower()
                if any(term in ingredients for term in ['meat', 'fish', 'chicken', 'beef', 'pork', 'cream', 'milk']):
                    return {'dietary_category': 'non-vegetarian', 'confidence': 0.9}
                elif any(term in ingredients for term in ['honey', 'eggs', 'cheese', 'butter']):
                    return {'dietary_category': 'vegetarian', 'confidence': 0.8}
                else:
                    return {'dietary_category': 'vegan', 'confidence': 0.7}
        return MockRuleModel()
    
    def _create_mock_roberta_model(self):
        """Create mock RoBERTa model for demonstration"""
        class MockRobertaModel:
            def predict(self, ingredients):
                ingredients = ingredients.lower()
                # Simplified logic - in reality this would be the trained transformer
                if len(ingredients.split(',')) < 3:
                    # Struggle with short lists
                    confidence = 0.6
                else:
                    confidence = 0.85
                
                if any(term in ingredients for term in ['meat', 'fish', 'chicken', 'cream']):
                    return {'dietary_category': 'non-vegetarian', 'confidence': confidence}
                elif 'honey' in ingredients:
                    return {'dietary_category': 'vegan', 'confidence': confidence}  # Common error
                else:
                    return {'dietary_category': 'vegan', 'confidence': confidence}
        return MockRobertaModel()
    
    def train(self, training_data_path='misclassification_analysis_results.json'):
        """Train the ensemble system"""
        print("Training IEEPL Classifier...")
        
        # Prepare training data
        training_data = self.reliability_predictor.prepare_training_data(training_data_path)
        
        # Train reliability predictor
        self.reliability_predictor.train(training_data)
        
        self.is_trained = True
        print("Training complete!")
    
    def predict(self, ingredients_text):
        """Make prediction using intelligent ensemble"""
        if not self.is_trained:
            print("Warning: Model not trained. Training now...")
            # Train with sample data
            self.train()
        
        # Get base model predictions
        rule_pred = self.rule_model.predict(ingredients_text)
        roberta_pred = self.roberta_model.predict(ingredients_text)
        
        # Predict reliability
        rule_reliability, roberta_reliability, features = self.reliability_predictor.predict_reliability(ingredients_text)
        
        # Resolve conflicts
        final_pred, resolution_method = self.conflict_resolver.resolve_conflict(
            rule_pred, roberta_pred, features, rule_reliability, roberta_reliability
        )
        
        # Calculate overall confidence
        confidence = max(rule_reliability, roberta_reliability)
        if final_pred == "requires_review":
            confidence = 0.0
        
        return {
            'prediction': final_pred,
            'confidence': confidence,
            'rule_prediction': rule_pred['dietary_category'],
            'roberta_prediction': roberta_pred['dietary_category'],
            'rule_reliability': rule_reliability,
            'roberta_reliability': roberta_reliability,
            'resolution_method': resolution_method,
            'requires_review': final_pred == "requires_review",
            'features': features
        }
    
    def explain_prediction(self, ingredients_text):
        """Provide detailed explanation of prediction"""
        result = self.predict(ingredients_text)
        
        explanation = f"""
Prediction Explanation for: "{ingredients_text}"
=====================================================

Base Model Predictions:
- Rule-Based: {result['rule_prediction']} (reliability: {result['rule_reliability']:.3f})
- RoBERTa: {result['roberta_prediction']} (reliability: {result['roberta_reliability']:.3f})

Final Decision: {result['prediction']}
Resolution Method: {result['resolution_method']}
Overall Confidence: {result['confidence']:.3f}

Key Features:
- Ingredient Count: {result['features']['ingredient_count']}
- Has Processing Terms: {result['features']['has_processing_terms']}
- Has Cultural Markers: {result['features']['has_cultural_markers']}
- Complexity Score: {result['features']['complexity_score']:.2f}
- Text Length: {result['features']['text_length']}

Requires Human Review: {result['requires_review']}
"""
        return explanation

def run_demonstration():
    """Demonstrate the IEEPL classifier"""
    print("=" * 60)
    print("IEEPL Classifier Demonstration")
    print("=" * 60)
    
    # Initialize and train classifier
    classifier = IEEPLClassifier()
    classifier.train()
    
    # Test cases from Checkpoint 4 analysis
    test_cases = [
        "pea protein isolate, natural flavors, guar gum",  # Rule-based struggles
        "cream, salt",  # RoBERTa struggles with short list
        "honey, oats, almonds",  # Disagreement case
        "mono- and diglycerides, natural flavors",  # Ambiguous case
        "nutritional yeast, cashew cream, truffle oil",  # Novel ingredients
        "anchovies, sea salt, water",  # Cultural food
    ]
    
    print("\nTesting IEEPL Classifier:")
    print("-" * 60)
    
    for ingredients in test_cases:
        print(f"\nInput: {ingredients}")
        result = classifier.predict(ingredients)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['resolution_method']}")
        if result['requires_review']:
            print("⚠️  FLAGGED FOR HUMAN REVIEW")
        print("-" * 40)
    
    # Detailed explanation for one case
    print("\n" + "=" * 60)
    print("DETAILED EXPLANATION EXAMPLE")
    print("=" * 60)
    print(classifier.explain_prediction("pea protein isolate, natural flavors, guar gum"))

if __name__ == "__main__":
    run_demonstration() 