# 📄 LabelSense Data README

## 1. Overview
**LabelSense** is an AI-driven food label analyzer that empowers consumers to understand product ingredients, dietary suitability, and allergen risks directly from packaging images. By combining OCR for text extraction and advanced NLP (including rule-based methods and LLM assistance) for ingredient parsing, LabelSense classifies vegan/vegetarian compliance, detects personalized allergens, and provides clear ingredient definitions—across multiple languages and product types.

The original dataset powering LabelSense is the **Open Food Facts (OFF) data dump** (~1 GB CSV) available at [https://world.openfoodfacts.org/data](https://world.openfoodfacts.org/data).  
The file `food_fr.csv` is a **curated subset** of that larger dataset, focusing on products sold in France, and containing only the fields relevant to the LabelSense pipeline:
- OCR input (ingredient images/text)  
- NLP targets (ingredients, dietary/allergen tags)  
- Minimal metadata for UI and evaluation

---

## 2. Source & License
- **Origin:** Open Food Facts (France region)  
- **Full data dump:** https://world.openfoodfacts.org/data  
- **License:** Open Database License (ODbL)  
- **Hugging Face mirror:** https://huggingface.co/datasets/openfoodfacts/product-database

---

## 3. File Details
- **Filename:** `food_fr.csv`  
- **Format:** UTF-8 CSV  
- **Rows:** 10,000 products  
- **Columns:** Selected subset relevant to LabelSense

---

## 4. Selected Columns

### Core Fields

| Column Name               | Type    | Purpose                                                                                   |
|:--------------------------|:--------|:------------------------------------------------------------------------------------------|
| `code`                    | String  | EAN/UPC barcode or internal product code (unique identifier).                             |
| `product_name`            | String  | Official product name as printed on packaging.                                            |
| `url`                     | String  | Link to the Open Food Facts product page (for reference or UI).                           |
| `categories_tags`         | String  | Taxonomy tags for product category (e.g. `en:sodas`, `fr:biscuits`), used for filtering. |
| `image_ingredients_url`   | String  | URL of the photo containing the ingredient list—**OCR input**.                            |
| `ingredients_text`        | String  | Raw ingredient list text—**NLP input/verification**.                                      |
| `ingredients_tags`        | String  | Normalized ingredient identifiers—**NER training labels**.                                |
| `labels_tags`             | String  | Dietary/certification tags (e.g. `en:vegan`)—**diet classifier**.                         |
| `allergens`               | String  | Declared allergens (e.g. `en:milk`)—**allergen detection**.                              |
| `traces`                  | String  | Possible traces of allergens—**allergen risk flags**.                                     |

### Extended Fields

| Column Name                   | Type    | Purpose                                                                                                  |
|:------------------------------|:--------|:--------------------------------------------------------------------------------------------------------|
| `serving_size`                | String  | Normalize per-serving nutritional values and display to users (e.g. “30 g”).                             |
| `nutrient_levels_tags`        | String  | Front-of-pack nutrition summary (e.g. `en:low-fat`, `en:high-salt`) for quick health-risk flags.         |
| `energy_100g`                 | Float   | Total energy (kcal) — for health score calculations.                                                     |
| `fat_100g`                    | Float   | Fat content — for dietary advice (e.g. “high-fat ingredients”).                                           |
| `saturated-fat_100g`          | Float   | Saturated fat — key for cardiovascular risk alerts.                                                      |
| `carbohydrates_100g`          | Float   | Carbs — useful for “low-carb” filtering or warnings.                                                     |
| `sugars_100g`                 | Float   | Sugars — for sugar-sensitivity profiles or “added sugar” warnings.                                      |
| `fiber_100g`                  | Float   | Fiber — can recommend higher-fiber alternatives.                                                         |
| `proteins_100g`               | Float   | Protein — helpful for athletes or high-protein diets.                                                    |
| `salt_100g`                   | Float   | Salt — for hypertension/cardiac dietary flags.                                                           |
| `additives_tags`              | String  | E-numbers and other additives (e.g. `en:e160a`)—flag undesired chemicals.                                |
| `ingredients_from_palm_oil_n` | Int     | Count of ingredients derived from palm oil—support sustainability badges.                                 |
| `brands`                      | String  | Brand filtering or brand-specific analyses.                                                              |
| `countries`                   | String  | Market region—test multilingual OCR or region-specific regulations.                                      |
| `labels`                      | String  | Human-readable certifications (e.g. “Bio”) for UI badges.                                                |
| `states_tags`                 | String  | Data quality/status tags (e.g. `en:confirmed`)—filter out poor-quality entries.                          |

---

## 5. Usage in LabelSense Pipeline

1. **OCR Stage**  
   - Preprocess and download images from `image_ingredients_url`.  
   - Use OCR (e.g., Tesseract) to extract text; compare against `ingredients_text` for evaluation.

2. **Ingredient Parsing & NER**  
   - Tokenize OCR output and align with `ingredients_tags` to train/fine-tune NER models.

3. **Dietary Classification**  
   - Use `labels_tags` to supervise vegan/vegetarian classification.

4. **Allergen Detection**  
   - Match parsed ingredients against `allergens` and `traces` to flag user allergens.

5. **UI & Metadata**  
   - Show `product_name`, link to `url`, and display categories via `categories_tags`.

---

## 6. Loading Example (Python/pandas)
```python
import pandas as pd

use_cols = [
    'code','product_name','url','categories_tags',
    'image_ingredients_url','ingredients_text','ingredients_tags',
    'labels_tags','allergens','traces',
    'serving_size','nutrient_levels_tags',
    'energy_100g','fat_100g','saturated-fat_100g',
    'carbohydrates_100g','sugars_100g','fiber_100g',
    'proteins_100g','salt_100g',
    'additives_tags','ingredients_from_palm_oil_n',
    'brands','countries','labels','states_tags'
]
df = pd.read_csv('food_fr.csv', usecols=[c for c in use_cols if c in pd.read_csv('food_fr.csv', nrows=0).columns])
```

---

## 7. Citation
If you use this data in your work, please cite:  
> Open Food Facts contributors. (2025). Open Food Facts database. https://world.openfoodfacts.org/data

---

### Contact
For questions about this dataset or the LabelSense project, contact:
- Khushboo Harsh Patel (kp3329@drexel.edu)  
- Shobhit Dixit (sd3733@drexel.edu)  
- Neel Rakeshbhai Patel (np928@drexel.edu)  
