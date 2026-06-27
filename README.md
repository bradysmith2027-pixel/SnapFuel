# 🍽️ SnapFuel

**Snap your food. Fuel your goals.**

SnapFuel is an AI-powered food recognition and calorie estimation app built with computer vision. Upload a photo of your food, enter how many grams you have, and SnapFuel instantly identifies the food and calculates your calorie intake — no manual logging required.

🔗 **[Try the Live Demo on Hugging Face](https://huggingface.co/spaces/Bman321/SnapFuel)**

📓 **[View the Model Training Notebook](notebooks/snapfuel_training.ipynb)** — Full pipeline including data preprocessing, EfficientNetB0 fine-tuning, and evaluation

---

## 🚀 What It Does

Traditional calorie tracking requires users to manually search for foods, estimate portions, and log every meal — a process that research consistently shows leads to high dropout rates over time. SnapFuel reduces this to two steps:

1. **Upload a photo** of your food
2. **Enter the amount** in grams

SnapFuel's model identifies the food, matches it to USDA nutritional data, and calculates your total calorie intake automatically.

---

## 🧠 How It Works

### Model Architecture
- **Backbone:** EfficientNetB0 pretrained on ImageNet
- **Dataset:** Food-101 (101 food categories, 1,000 images each)
- **Training Pipeline:** Two-phase transfer learning
  - **Phase 1 – Feature Extraction:** Frozen EfficientNet weights, only the classification head is trained on Food-101 classes
  - **Phase 2 – Fine-Tuning:** Last 30 layers unfrozen and retrained at a lower learning rate with early stopping to prevent overfitting
- **Output:** Softmax activation across 101 food classes, returning top-3 predictions with confidence scores
- **Input Size:** 160x160 pixels

### Performance
| Metric | Score |
|---|---|
| Training Accuracy | ~50% |
| Validation Accuracy | ~63% |
| Top-3 Prediction | Available |

### Calorie Estimation
- Food predictions are matched against a custom-built calorie database sourced from the **USDA FoodData Central API**
- Calories are scaled to the user's entered gram amount using: `(user_grams / serving_size_g) × calories_per_serving`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | EfficientNetB0 (TensorFlow / Keras) |
| Dataset | Food-101 |
| Nutrition Data | USDA FoodData Central API |
| Frontend / UI | Gradio |
| Deployment | Hugging Face Spaces |
| Language | Python |

---

## 📁 Project Structure

```
snapfuel/
├── app.py                  # Main Gradio app and prediction logic
├── class_names.json        # 101 Food-101 class labels
├── calorie_database.json   # USDA-sourced calorie data for each class
├── requirements.txt        # Python dependencies
└── README.md
```

> **Note:** The trained model file (`food_recognition_efficientnetb0_fine_tuned.h5`) is hosted on Hugging Face Spaces due to its file size and is not included in this repository.

---

## ⚙️ Running Locally

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# Clone the repo
git clone https://github.com/yourusername/snapfuel.git
cd snapfuel

# Install dependencies
pip install -r requirements.txt

# Download the model from Hugging Face and place it in the root directory
# Model: food_recognition_efficientnetb0_fine_tuned.h5

# Launch the app
python app.py
```

---

## 🔬 Research Background

SnapFuel was built with a foundation in healthcare research:

- **Burke et al. (2011)** found that dietary self-monitoring was positively associated with weight loss in all 15 studies reviewed, reaching the highest level of clinical evidence (Class IIa, Level A)
- **Wang et al. (2024)** identified manual data entry and cognitive load as the primary drivers of dropout in digital dietary interventions, and specifically recommended image processing as a solution
- SnapFuel directly addresses these findings by replacing manual food lookup with a two-step photo-based flow

---

## ⚠️ Limitations & Future Improvements

- Limited to **101 food categories** from the Food-101 dataset — expanding this is the most impactful next step
- **Portion size must be manually entered** — automatic volume estimation from images is a future goal
- Performance drops with poor lighting, extreme angles, or mixed dishes
- Validation accuracy of ~63% could be improved with a larger, better-regulated training dataset and higher resolution images
- A mobile app interface would significantly improve accessibility and user experience

---

## 📚 References

- Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101. ECCV.
- Burke, L.E., Wang, J., & Sevick, M.A. (2011). Self-monitoring in weight loss. *Journal of the American Dietetic Association.*
- Wang, Y. et al. (2024). Framework development for reducing attrition in digital dietary interventions. *Journal of Medical Internet Research.*
- Liu, H., Xie, Z., & Or, C. (2024). Willingness to pay for health apps. *Digital Health.*
