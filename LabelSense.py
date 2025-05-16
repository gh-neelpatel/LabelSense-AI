# LabelSense: AI-Driven Food Label Analyzer
# An NLP Class Project Implementation

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import re
import os
import requests
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import warnings
import sys
warnings.filterwarnings('ignore')
from google.colab import files

# Install tesseract in Colab environment
if 'google.colab' in sys.modules:
    import subprocess
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "tesseract-ocr"], check=True)
    subprocess.run(["apt-get", "install", "-y", "tesseract-ocr-eng", "tesseract-ocr-fra", "tesseract-ocr-spa"], check=True)
    subprocess.run(["pip", "install", "pytesseract"], check=True)
    # Set path for Colab
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:
    # Set path for local Windows environment
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load('en_core_web_sm')

# Constants
ALLERGENS = [
    'milk', 'dairy', 'eggs', 'egg', 'peanuts', 'peanut', 'tree nuts', 'almond', 'almonds', 'walnut', 'walnuts', 
    'cashew', 'cashews', 'hazelnut', 'hazelnuts', 'shellfish', 'fish', 'wheat', 'gluten', 'soy', 'soya', 
    'sesame', 'mustard', 'celery', 'lupin', 'sulphites', 'sulfites', 'molluscs', 'mollusks'
]

NON_VEGAN = [
    'milk', 'dairy', 'cheese', 'eggs', 'egg', 'honey', 'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 
    'gelatin', 'gelatine', 'lard', 'tallow', 'whey', 'casein', 'lactose', 'rennet', 'shellac', 'carmine', 
    'isinglass', 'albumin', 'cochineal', 'fish', 'shellfish', 'beef fat', 'butter', 'buttermilk', 'yogurt', 
    'cream', 'mayonnaise', 'bacon', 'duck', 'goose'
]

NON_VEGETARIAN = [
    'meat', 'beef', 'pork', 'chicken', 'turkey', 'lamb', 'gelatin', 'gelatine', 'lard', 'tallow', 'rennet',
    'shellac', 'isinglass', 'carmine', 'cochineal', 'fish', 'shellfish', 'beef fat', 'bacon', 'duck', 'goose'
]

# Class to handle image preprocessing
class ImagePreprocessor:
    @staticmethod
    def resize_image(image, width=800):
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        ratio = width / w
        dim = (width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def denoise_image(image):
        """Apply denoising to the image"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    @staticmethod
    def apply_thresholding(image):
        """Apply thresholding to prepare for OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    @staticmethod
    def detect_orientation(image):
        """Detect and correct image orientation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Tesseract OSD to detect orientation
        osd = pytesseract.image_to_osd(gray)
        angle = int(re.search(r'(?<=Rotate: )\d+', osd).group())
        
        if angle != 0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
        return image
    
    @staticmethod
    def enhance_contrast(image):
        """Enhance contrast using CLAHE"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def preprocess(image):
        """Apply full preprocessing pipeline"""
        if image is None:
            raise ValueError("Image not loaded properly")
        
        try:
            # Apply preprocessing steps
            image = ImagePreprocessor.resize_image(image)
            image = ImagePreprocessor.denoise_image(image)
            image = ImagePreprocessor.enhance_contrast(image)
            
            # Try to detect and correct orientation
            try:
                image = ImagePreprocessor.detect_orientation(image)
            except:
                pass  # Skip if orientation detection fails
            
            # Create a binary version for OCR
            binary = ImagePreprocessor.apply_thresholding(image)
            
            return image, binary
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return image, None

# Class to handle OCR operations
class OCREngine:
    @staticmethod
    def extract_text(image, languages=None):
        """Extract text from the image using Tesseract OCR
        
        Args:
            image: The image to process
            languages: Language codes to use (e.g., 'eng', 'fra', 'spa', 'eng+fra+spa')
        """
        if languages is None:
            languages = 'eng+fra+spa'  # Default to all supported languages
            
        try:
            # Extract using PSM 4 (assume a single column of text) with multiple languages
            config = f'--psm 4 -l {languages}'
            text = pytesseract.image_to_string(image, config=config)
            
            # If little text is found, try different PSM modes
            if len(text) < 50:
                # Try different PSM modes to get the best results
                text_psm_6 = pytesseract.image_to_string(image, config=f'--psm 6 -l {languages}')
                text_psm_11 = pytesseract.image_to_string(image, config=f'--psm 11 -l {languages}')
                text_psm_3 = pytesseract.image_to_string(image, config=f'--psm 3 -l {languages}')
                
                # Use the one with most text
                candidates = [text, text_psm_3, text_psm_6, text_psm_11]
                text = max(candidates, key=len)
            
            # Clean up the extracted text
            text = text.replace('\n\n', '\n').strip()
            return text
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""
    
    @staticmethod
    def extract_ingredients_section(text):
        """Extract the ingredients section from the OCR text"""
        # Various ways ingredients sections are labeled
        patterns = [
            r'(?i)ingredients\s*:(.+?)(?:\.|$|\n\n|\*{3})',
            r'(?i)ingredients\s*list\s*:(.+?)(?:\.|$|\n\n|\*{3})',
            r'(?i)contains\s*:(.+?)(?:\.|$|\n\n|\*{3})',
            r'(?i)ingred(?:i|l)ents\s*:(.+?)(?:\.|$|\n\n|\*{3})',
            r'(?i)ingredients?[\s\.\)](.+?)(?:\.|$|\n\n|\*{3})',
            r'(?i)ingredients(.+?)(?=allerg[ye]|$|\n\n|\*{3})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                if len(ingredients_text) > 20:  # Only return if we got a substantial match
                    return ingredients_text
        
        # If the text is short, it might be just the ingredients section itself
        if len(text) < 1000 and ',' in text:
            return text
            
        # If no specific ingredients section is found, check if the whole text contains common ingredients
        words = text.lower().split()
        ingredient_markers = ['sugar', 'salt', 'flour', 'milk', 'butter', 'egg', 'oil']
        if any(marker in words for marker in ingredient_markers):
            return text
        
        return text

# Class to analyze ingredients
class IngredientAnalyzer:
    def __init__(self):
        # Initialize ingredient database with Open Food Facts data
        # In a real implementation, you would load this data from a database
        self.ingredient_database = {
            # Common ingredients and their properties
            # Format: 'ingredient': {'vegan': bool, 'vegetarian': bool, 'allergen': bool}
        }
        
        # Load pre-defined lists
        self.allergens = set(ALLERGENS)
        self.non_vegan = set(NON_VEGAN)
        self.non_vegetarian = set(NON_VEGETARIAN)
        
        # Add cake-specific ingredients to the lists
        self.allergens.update(['wheat', 'soya', 'sulphur dioxide', 'sulphites', 'sulfites'])
        self.non_vegan.update(['milk powder', 'skimmed milk powder', 'milk powder', 'belgian chocolate', 'cocoa butter'])
        
        # Try to load the dietary classifier model if available
        try:
            from DietaryClassifier import DietaryClassifier, SimpleRuleBasedClassifier
            model_path = 'models/dietary_classifier_roberta.pt'
            if os.path.exists(model_path):
                self.dietary_classifier = DietaryClassifier(model_path)
                print("Loaded dietary classification model")
            else:
                self.dietary_classifier = SimpleRuleBasedClassifier()
                print("Using rule-based dietary classifier (model not found)")
        except ImportError as e:
            print(f"Dietary classifier not available: {e}")
            self.dietary_classifier = None
    
    def parse_ingredients(self, ingredients_text):
        """Parse the ingredients text into a list of individual ingredients"""
        if not ingredients_text:
            return []
        
        # Clean up formatting
        text = re.sub(r'\s+', ' ', ingredients_text)
        
        # Try to split by common separators while preserving parenthetical content
        # First, extract all text in parentheses and replace with placeholders
        parenthesis_contents = []
        def replace_parens(match):
            parenthesis_contents.append(match.group(1))
            return f"__PAREN_{len(parenthesis_contents)-1}__"
        
        # Replace parentheses content with placeholders
        processed_text = re.sub(r'\(([^)]+)\)', replace_parens, text)
        
        # Split by commas and other separators
        raw_ingredients = re.split(r',|\*|•|;', processed_text)
        ingredients = []
        
        # Process each raw ingredient
        for raw in raw_ingredients:
            raw = raw.strip().lower()
            if not raw:
                continue
                
            # Replace parenthesis placeholders with original content
            while '__PAREN_' in raw:
                match = re.search(r'__PAREN_(\d+)__', raw)
                if match:
                    idx = int(match.group(1))
                    if idx < len(parenthesis_contents):
                        raw = raw.replace(f"__PAREN_{idx}__", f"({parenthesis_contents[idx]})")
            
            ingredients.append(raw)
            
            # Also add the parenthetical parts as separate ingredients
            for paren_content in parenthesis_contents:
                sub_ingredients = re.split(r',|\*|•|;', paren_content)
                for sub in sub_ingredients:
                    sub = sub.strip().lower()
                    if sub and len(sub) > 2 and sub not in ingredients:
                        ingredients.append(sub)
        
        return ingredients
    
    def is_vegan(self, ingredients):
        """Check if the ingredient list is vegan"""
        if not ingredients:
            return {"is_vegan": False, "non_vegan_ingredients": ["Unknown ingredients"]}
        
        # Use the custom dietary classifier if available
        if hasattr(self, 'dietary_classifier') and self.dietary_classifier is not None:
            # Join ingredients into a single string
            ingredients_text = ", ".join(ingredients)
            
            # Get prediction from classifier
            try:
                prediction = self.dietary_classifier.predict(ingredients_text)
                
                # If it's vegan, we don't need to find the non-vegan ingredients
                if prediction['is_vegan']:
                    return {
                        "is_vegan": True,
                        "non_vegan_ingredients": [],
                        "model_used": "classifier"
                    }
            except Exception as e:
                print(f"Error using dietary classifier: {str(e)}")
                # Fall back to rule-based method
        
        # Fall back to the original rule-based method
        non_vegan_ingredients = []
        
        for ingredient in ingredients:
            # Check against non-vegan items
            for non_vegan_item in self.non_vegan:
                if non_vegan_item in ingredient.lower():
                    non_vegan_ingredients.append(ingredient)
                    break
        
        return {
            "is_vegan": len(non_vegan_ingredients) == 0,
            "non_vegan_ingredients": non_vegan_ingredients,
            "model_used": "rule-based"
        }
    
    def is_vegetarian(self, ingredients):
        """Check if the ingredient list is vegetarian"""
        if not ingredients:
            return {"is_vegetarian": False, "non_vegetarian_ingredients": ["Unknown ingredients"]}
        
        # Use the custom dietary classifier if available
        if hasattr(self, 'dietary_classifier') and self.dietary_classifier is not None:
            # Join ingredients into a single string
            ingredients_text = ", ".join(ingredients)
            
            # Get prediction from classifier
            try:
                prediction = self.dietary_classifier.predict(ingredients_text)
                
                # If it's vegetarian, we don't need to find the non-vegetarian ingredients
                if prediction['is_vegetarian']:
                    return {
                        "is_vegetarian": True,
                        "non_vegetarian_ingredients": [],
                        "model_used": "classifier"
                    }
            except Exception as e:
                print(f"Error using dietary classifier: {str(e)}")
                # Fall back to rule-based method
        
        # Fall back to the original rule-based method
        non_vegetarian_ingredients = []
        
        for ingredient in ingredients:
            # Check against non-vegetarian items
            for non_veg_item in self.non_vegetarian:
                if non_veg_item in ingredient.lower():
                    non_vegetarian_ingredients.append(ingredient)
                    break
        
        return {
            "is_vegetarian": len(non_vegetarian_ingredients) == 0,
            "non_vegetarian_ingredients": non_vegetarian_ingredients,
            "model_used": "rule-based"
        }
    
    def find_allergens(self, ingredients):
        """Identify potential allergens in the ingredient list"""
        if not ingredients:
            return {"allergens_found": False, "allergens": []}
        
        allergens_found = []
        
        for ingredient in ingredients:
            # Check against allergens list
            for allergen in self.allergens:
                if allergen in ingredient.lower():
                    allergens_found.append(ingredient)
                    break
        
        return {
            "allergens_found": len(allergens_found) > 0,
            "allergens": allergens_found
        }
    
    def explain_ingredient(self, ingredient):
        """Provide a simple explanation of an ingredient"""
        # This would ideally use a database or API to look up ingredients
        # For demonstration, we'll return a simple placeholder
        return f"Definition of {ingredient}: This is a placeholder explanation. In a real implementation, this would provide detailed information about the ingredient."

# Class to handle the complete analysis process
class LabelSense:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.analyzer = IngredientAnalyzer()
    
    def analyze_label(self, image_path, languages=None):
        """Analyze a food label image
        
        Args:
            image_path: Path to the image or image data
            languages: Language codes to use for OCR (e.g., 'eng', 'fra', 'spa', 'eng+fra+spa')
        """
        # Load the image
        try:
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                # Assume it's already a numpy array
                image = image_path
                
            if image is None:
                return {"error": "Could not load image"}
        except Exception as e:
            return {"error": f"Error loading image: {str(e)}"}
        
        # Preprocess the image
        try:
            processed_image, binary_image = self.preprocessor.preprocess(image)
        except Exception as e:
            return {"error": f"Error preprocessing image: {str(e)}"}
        
        # Extract text using OCR
        try:
            if binary_image is not None:
                text = self.ocr_engine.extract_text(binary_image, languages)
            else:
                text = self.ocr_engine.extract_text(processed_image, languages)
        except Exception as e:
            return {"error": f"OCR error: {str(e)}"}
        
        # Extract ingredients section
        ingredients_text = self.ocr_engine.extract_ingredients_section(text)
        
        # Parse ingredients
        ingredients = self.analyzer.parse_ingredients(ingredients_text)
        
        # Analyze ingredients
        vegan_results = self.analyzer.is_vegan(ingredients)
        vegetarian_results = self.analyzer.is_vegetarian(ingredients)
        allergen_results = self.analyzer.find_allergens(ingredients)
        
        # Compile results
        results = {
            "ingredients_detected": ingredients,
            "vegan_analysis": vegan_results,
            "vegetarian_analysis": vegetarian_results,
            "allergen_analysis": allergen_results,
            "full_text": text,
            "languages_used": languages or "eng+fra+spa"
        }
        
        return results

# Function to download a sample image for testing
def download_sample_image():
    """Download a sample food label image for testing"""
    # Updated URL to a more reliable source
    url = "https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/demo_text_ocr.jpg"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open("sample_label.jpg", "wb") as f:
            f.write(response.content)
        return "sample_label.jpg"
    else:
        print("Failed to download sample image")
        return None

# Main function to demonstrate the LabelSense system
def main():
    # Initialize LabelSense
    label_sense = LabelSense()
    
    # Download a sample image
    print("Downloading a sample food label image...")
    sample_image_path = download_sample_image()
    
    if sample_image_path:
        print(f"Sample image downloaded as {sample_image_path}")
        
        # Analyze the label
        print("Analyzing food label...")
        results = label_sense.analyze_label(sample_image_path)
        
        # Display results
        print("\n===== LabelSense Analysis Results =====")
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print("\nIngredients Detected:")
            for ingredient in results["ingredients_detected"]:
                print(f"- {ingredient}")
            
            print("\nVegan Analysis:")
            print(f"Is Vegan: {results['vegan_analysis']['is_vegan']}")
            if not results['vegan_analysis']['is_vegan']:
                print("Non-Vegan Ingredients:")
                for ingredient in results['vegan_analysis']['non_vegan_ingredients']:
                    print(f"- {ingredient}")
            
            print("\nVegetarian Analysis:")
            print(f"Is Vegetarian: {results['vegetarian_analysis']['is_vegetarian']}")
            if not results['vegetarian_analysis']['is_vegetarian']:
                print("Non-Vegetarian Ingredients:")
                for ingredient in results['vegetarian_analysis']['non_vegetarian_ingredients']:
                    print(f"- {ingredient}")
            
            print("\nAllergen Analysis:")
            print(f"Allergens Found: {results['allergen_analysis']['allergens_found']}")
            if results['allergen_analysis']['allergens_found']:
                print("Allergens:")
                for allergen in results['allergen_analysis']['allergens']:
                    print(f"- {allergen}")
    else:
        print("Could not download sample image. Please provide your own image path.")

# Example of running the analysis on a local image
def analyze_local_image(image_path):
    # Initialize LabelSense
    label_sense = LabelSense()
    
    # Analyze the label
    print(f"Analyzing food label from {image_path}...")
    results = label_sense.analyze_label(image_path)
    
    # Display results
    print("\n===== LabelSense Analysis Results =====")
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\nIngredients Detected:")
        for ingredient in results["ingredients_detected"]:
            print(f"- {ingredient}")
        
        print("\nVegan Analysis:")
        print(f"Is Vegan: {results['vegan_analysis']['is_vegan']}")
        if not results['vegan_analysis']['is_vegan']:
            print("Non-Vegan Ingredients:")
            for ingredient in results['vegan_analysis']['non_vegan_ingredients']:
                print(f"- {ingredient}")
        
        print("\nVegetarian Analysis:")
        print(f"Is Vegetarian: {results['vegetarian_analysis']['is_vegetarian']}")
        if not results['vegetarian_analysis']['is_vegetarian']:
            print("Non-Vegetarian Ingredients:")
            for ingredient in results['vegetarian_analysis']['non_vegetarian_ingredients']:
                print(f"- {ingredient}")
        
        print("\nAllergen Analysis:")
        print(f"Allergens Found: {results['allergen_analysis']['allergens_found']}")
        if results['allergen_analysis']['allergens_found']:
            print("Allergens:")
            for allergen in results['allergen_analysis']['allergens']:
                print(f"- {allergen}")

# Advanced functionality: Train a custom model for ingredient detection
def train_ingredient_detection_model():
    """Train a custom model to detect ingredients from label images"""
    # This is a placeholder for a real training function
    # In an actual implementation, this would:
    # 1. Load a dataset of food label images with annotated ingredient lists
    # 2. Extract features using computer vision techniques
    # 3. Train a neural network to identify ingredients
    
    # Example neural network architecture (not functional without proper dataset)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Number of classes would depend on the task
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model architecture defined for ingredient detection.")
    print("Note: This is a placeholder. Actual training requires a dataset.")
    
    return model

# Function to build an ingredient database from Open Food Facts
def build_ingredient_database():
    """Build an ingredient database from Open Food Facts data"""
    # In a real implementation, you would:
    # 1. Download the Open Food Facts data dump
    # 2. Extract ingredient information
    # 3. Build a database of ingredients with properties
    
    print("Building ingredient database from Open Food Facts...")
    print("Note: This is a placeholder. Actual database building requires downloading and processing the Open Food Facts dataset.")
    
    # Example of how you would process the data
    database = {}
    
    # Placeholder for a few sample entries
    database['sugar'] = {'vegan': True, 'vegetarian': True, 'allergen': False}
    database['milk'] = {'vegan': False, 'vegetarian': True, 'allergen': True}
    database['beef'] = {'vegan': False, 'vegetarian': False, 'allergen': False}
    
    print(f"Created a sample database with {len(database)} entries.")
    return database

def analyze_label_image(image_file):
    """Create a detailed analysis of the uploaded image"""
    # Initialize LabelSense
    label_sense = LabelSense()
    
    print("Which languages should be used for OCR?")
    print("1. English only")
    print("2. English + French")
    print("3. English + Spanish")
    print("4. All (English, French, Spanish)")
    
    try:
        choice = int(input("Enter choice (1-4, default is 4): ") or "4")
    except:
        choice = 4
        
    # Map choice to language codes
    lang_map = {
        1: "eng",
        2: "eng+fra",
        3: "eng+spa",
        4: "eng+fra+spa"
    }
    languages = lang_map.get(choice, "eng+fra+spa")
    
    # Analyze the label
    print(f"Analyzing food label image in detail using languages: {languages}...")
    results = label_sense.analyze_label(image_file, languages)
    
    # Display results with more detailed information
    print("\n===== LabelSense AI Detailed Analysis Results =====")
    print(f"Languages used for OCR: {results.get('languages_used', 'eng+fra+spa')}")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
        
    print("\n📋 EXTRACTED TEXT:")
    print("----------------")
    print(results["full_text"])
    
    print("\n🥘 INGREDIENTS DETECTED:")
    print("---------------------")
    if results["ingredients_detected"]:
        for i, ingredient in enumerate(results["ingredients_detected"], 1):
            print(f"{i}. {ingredient}")
    else:
        print("No specific ingredients could be detected with confidence.")
    
    print("\n🌱 DIETARY ANALYSIS:")
    print("-----------------")
    print(f"Vegan Status: {'✅ Suitable' if results['vegan_analysis']['is_vegan'] else '❌ Not suitable'} for vegans")
    if not results['vegan_analysis']['is_vegan'] and results['vegan_analysis']['non_vegan_ingredients']:
        print("Non-Vegan Ingredients:")
        for ingredient in results['vegan_analysis']['non_vegan_ingredients']:
            print(f"- {ingredient}")
    
    print(f"\nVegetarian Status: {'✅ Suitable' if results['vegetarian_analysis']['is_vegetarian'] else '❌ Not suitable'} for vegetarians")
    if not results['vegetarian_analysis']['is_vegetarian'] and results['vegetarian_analysis']['non_vegetarian_ingredients']:
        print("Non-Vegetarian Ingredients:")
        for ingredient in results['vegetarian_analysis']['non_vegetarian_ingredients']:
            print(f"- {ingredient}")
    
    print("\n⚠️ ALLERGEN INFORMATION:")
    print("----------------------")
    print(f"Allergens Found: {'⚠️ Yes' if results['allergen_analysis']['allergens_found'] else '✅ None detected'}")
    if results['allergen_analysis']['allergens_found']:
        print("Potential Allergens:")
        for allergen in results['allergen_analysis']['allergens']:
            print(f"- {allergen}")
            
    print("\n🔍 ADDITIONAL INFORMATION:")
    print("------------------------")
    print("This analysis is based on automated OCR and should not replace reading the actual label.")
    print("If you have allergies or dietary restrictions, please consult the original packaging.")

# Main execution block at the end
if __name__ == "__main__":
    print("LabelSense AI Food Label Analyzer")
    print("=================================")
    
    if 'google.colab' in sys.modules:
        # Import IPython for interactive widgets
        from IPython.display import display
        import ipywidgets as widgets
        
        # Create language selection widget
        language_select = widgets.RadioButtons(
            options=[
                ('English only', 'eng'),
                ('English + French', 'eng+fra'),
                ('English + Spanish', 'eng+spa'),
                ('All (English, French, Spanish)', 'eng+fra+spa')
            ],
            value='eng+fra+spa',
            description='Select OCR Languages:',
            disabled=False
        )
        
        display(language_select)
        print("Upload a food label image to analyze:")
        
        # Upload the image
        uploaded = files.upload()
        
        if uploaded:
            image_path = list(uploaded.keys())[0]
            selected_languages = language_select.value
            
            # Initialize LabelSense
            label_sense = LabelSense()
            
            # Analyze the label
            print(f"Analyzing food label image using languages: {selected_languages}...")
            results = label_sense.analyze_label(image_path, selected_languages)
            
            # Display results with more detailed information
            print("\n===== LabelSense AI Detailed Analysis Results =====")
            print(f"Languages used for OCR: {results.get('languages_used', 'eng+fra+spa')}")
            
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print("\n📋 EXTRACTED TEXT:")
                print("----------------")
                print(results["full_text"])
                
                print("\n🥘 INGREDIENTS DETECTED:")
                print("---------------------")
                if results["ingredients_detected"]:
                    for i, ingredient in enumerate(results["ingredients_detected"], 1):
                        print(f"{i}. {ingredient}")
                else:
                    print("No specific ingredients could be detected with confidence.")
                
                print("\n🌱 DIETARY ANALYSIS:")
                print("-----------------")
                print(f"Vegan Status: {'✅ Suitable' if results['vegan_analysis']['is_vegan'] else '❌ Not suitable'} for vegans")
                if not results['vegan_analysis']['is_vegan'] and results['vegan_analysis']['non_vegan_ingredients']:
                    print("Non-Vegan Ingredients:")
                    for ingredient in results['vegan_analysis']['non_vegan_ingredients']:
                        print(f"- {ingredient}")
                
                print(f"\nVegetarian Status: {'✅ Suitable' if results['vegetarian_analysis']['is_vegetarian'] else '❌ Not suitable'} for vegetarians")
                if not results['vegetarian_analysis']['is_vegetarian'] and results['vegetarian_analysis']['non_vegetarian_ingredients']:
                    print("Non-Vegetarian Ingredients:")
                    for ingredient in results['vegetarian_analysis']['non_vegetarian_ingredients']:
                        print(f"- {ingredient}")
                
                print("\n⚠️ ALLERGEN INFORMATION:")
                print("----------------------")
                print(f"Allergens Found: {'⚠️ Yes' if results['allergen_analysis']['allergens_found'] else '✅ None detected'}")
                if results['allergen_analysis']['allergens_found']:
                    print("Potential Allergens:")
                    for allergen in results['allergen_analysis']['allergens']:
                        print(f"- {allergen}")
                        
                print("\n🔍 ADDITIONAL INFORMATION:")
                print("------------------------")
                print("This analysis is based on automated OCR and should not replace reading the actual label.")
                print("If you have allergies or dietary restrictions, please consult the original packaging.")
    else:
        # Option 1: Run with sample image download
        main()
        
        # Option 2: Run with your own image (replace with your image path)
        # my_image_path = "path/to/your/food_label.jpg"  # Change this to your image path
        # analyze_local_image(my_image_path)