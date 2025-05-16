# LabelSense.AI

## Overview
LabelSense.AI is an AI-driven food label analyzer that empowers consumers to understand product ingredients, dietary suitability (vegan/vegetarian/non-vegetarian), and allergen risks directly from packaging images. It combines OCR, NLP, and ML models to extract, parse, and classify food label information, supporting English, French, and Spanish.

## Features
- Multilingual OCR (English, French, Spanish)
- Ingredient extraction and NER
- Dietary classification (vegan, vegetarian, non-vegetarian)
- Allergen detection (rule-based + LLM)
- Ingredient explanation (LLM-powered)
- Personalization logic for user-specific allergens
- Evaluation pipeline with quantitative and human feedback metrics

## Setup
1. Clone the repo:
   ```bash
   git clone <your_repo_url>
   cd LabelSense.AI
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords
   ```
4. Download the Open Food Facts dataset and place `food_fr.csv` in the project directory (see Checkpoint 2 for details).

## Usage
- **Run the main pipeline:**
  ```bash
  python LabelSense.py
  ```
- **Analyze a local image:**
  ```python
  from LabelSense import LabelSense
  label_sense = LabelSense()
  results = label_sense.analyze_label('path/to/your/image.jpg')
  print(results)
  ```
- **Train or test models:**
  See the provided Jupyter notebooks or scripts for NER and dietary classification experiments.

## Results & Analysis
- See the `Ckpt 4/` folder for results tables and analysis.
- Evaluation metrics and human feedback are summarized in `Ckpt 3/ckpt1_updated.md` and `Ckpt 4/`.

## Project Structure
```
LabelSense.AI/
├── LabelSense.py
├── requirements.txt
├── README.md
├── food_fr.csv (not included)
├── Ckpt 1/ (proposal)
├── ckpt 2/ (dataset & data README)
├── Ckpt 3/ (evaluation & metrics)
├── Ckpt 4/ (results & analysis)
├── notebooks/ (optional: experiments)
```

## References
- [Open Food Facts](https://world.openfoodfacts.org/data)
- [TensorFlow custom training walkthrough](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
- [MiniGPT-4 local setup](https://github.com/rbbrdckybk/MiniGPT-4)

## Checkpoints
- Proposal: `Ckpt 1/LabelSense_Proposal_higlighted_updates.pdf`
- Data: `ckpt 2/README_LabelSense_UpdatedProposal.md`
- Evaluation: `Ckpt 3/ckpt1_updated.md`
- Results: `Ckpt 4/` 