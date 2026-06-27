import json
import os
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from typing import Dict

MODEL_PATH = "food_recognition_efficientnetb0_fine_tuned.h5"
CLASS_NAMES_PATH = "class_names.json"
CALORIE_DB_PATH = "calorie_database.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please ensure it's uploaded.")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}. Please ensure it's uploaded.")

if not os.path.exists(CALORIE_DB_PATH):
    raise FileNotFoundError(f"Calorie database file not found: {CALORIE_DB_PATH}. Please ensure it's uploaded.")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

with open(CALORIE_DB_PATH, "r") as f:
    calorie_database = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

IMG_HEIGHT = 160
IMG_WIDTH = 160

def predict_food_and_calories(image_path: str) -> Dict:
    try:
        img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_food = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100

        top_3_idx = np.argsort(predictions)[::-1][:3]
        top_3_predictions = [
            {
                "food": class_names[int(idx)],
                "confidence": float(predictions[int(idx)]) * 100
            }
            for idx in top_3_idx
        ]

        calorie_info = calorie_database.get(predicted_food, {})

        return {
            "predicted_food": predicted_food,
            "confidence": f"{confidence:.2f}%",
            "calories_per_serving": calorie_info.get("calories_per_serving", "N/A"),
            "serving_size_g": calorie_info.get("serving_size_g", "N/A"),
            "usda_food_name": calorie_info.get("usda_food_name", "N/A"),
            "top_3_predictions": top_3_predictions
        }

    except Exception as e:
        return {
            "predicted_food": "Error",
            "confidence": "N/A",
            "calories_per_serving": "N/A",
            "serving_size_g": "N/A",
            "usda_food_name": str(e),
            "top_3_predictions": []
        }

def calculate_total_calories(calories_per_serving, serving_size_g, amount_g):
    """Calculate total calories based on user input amount"""
    try:
        if calories_per_serving == "N/A" or serving_size_g == "N/A":
            return "N/A", "N/A"
        
        calories_per_serving = float(calories_per_serving)
        serving_size_g = float(serving_size_g)
        amount_g = float(amount_g)
        
        if amount_g <= 0:
            return "N/A", "Invalid amount"
        
        # Calculate calories per gram
        calories_per_gram = calories_per_serving / serving_size_g
        # Calculate total calories for the user's amount
        total_calories = calories_per_gram * amount_g
        
        return total_calories, f"{amount_g}g"
    except (ValueError, ZeroDivisionError, TypeError):
        return "N/A", "Invalid input"

def format_prediction_output(result: Dict, amount_g: float = None) -> str:
    if result["predicted_food"] == "Error":
        return f"""
<div class="result-card error-card">
    <h2>Prediction Error</h2>
    <p>An error occurred during prediction.</p>
    <p>{result['usda_food_name']}</p>
</div>
"""

    predicted_food = result["predicted_food"].replace("_", " ").title()

    output_md = f"""
<div class="result-card">
    <h2>🍽️ Predicted Food</h2>
    <div class="food-name">{predicted_food}</div>

    <div class="confidence-box">
        <span>Confidence</span>
        <strong>{result['confidence']}</strong>
    </div>

    <h3>🔥 Calorie Information</h3>
    <ul>
        <li><strong>USDA Matched Name:</strong> {result['usda_food_name']}</li>
        <li><strong>Calories per serving:</strong> {result['calories_per_serving']} kcal</li>
        <li><strong>Serving size:</strong> {result['serving_size_g']}g</li>
"""

    # Add personalized calorie calculation if amount is provided
    if amount_g and amount_g > 0:
        total_cal, amount_display = calculate_total_calories(
            result['calories_per_serving'],
            result['serving_size_g'],
            amount_g
        )
        if total_cal != "N/A":
            output_md += f"""        <li><strong style="color: #ef4444;">Your Amount ({amount_display}):</strong> <span style="font-size: 20px; color: #ef4444; font-weight: 900;">{total_cal:.0f} kcal</span></li>
"""

    output_md += """    </ul>

    <h3>🏆 Top 3 Predictions</h3>
    <ol>
"""

    if not result["top_3_predictions"]:
        output_md += "<li>No top predictions available.</li>"
    else:
        for pred in result["top_3_predictions"]:
            food_name = pred["food"].replace("_", " ").title()
            output_md += f"<li><strong>{food_name}</strong> — {pred['confidence']:.2f}%</li>"

    output_md += """
    </ol>
</div>
"""
    return output_md

def gradio_predict_wrapper(image_path: str, amount_g: float = None) -> str:
    if image_path is None:
        return """
<div class="result-card">
    <h2>Upload an image</h2>
    <p>Please upload a food image to get a prediction.</p>
</div>
"""
    prediction_result = predict_food_and_calories(image_path)
    return format_prediction_output(prediction_result, amount_g)

custom_css = """
body {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 45%, #fed7aa 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: transparent !important;
}

#snapfuel-header {
    text-align: center;
    padding: 28px 20px 10px 20px;
}

#snapfuel-header h1 {
    font-size: 52px;
    font-weight: 900;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #f97316, #ef4444, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#snapfuel-header p {
    font-size: 18px;
    color: #7c2d12 !important;
    margin-top: 0;
}

#main-card {
    background: rgba(255, 255, 255, 0.88);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 20px 45px rgba(234, 88, 12, 0.18);
    border: 1px solid rgba(251, 146, 60, 0.25);
}

.result-card {
    background: #ffffff !important;
    border-radius: 22px;
    padding: 24px;
    border: 1px solid #fed7aa;
    box-shadow: 0 12px 28px rgba(249, 115, 22, 0.12);
    color: #2b1205 !important;
    width: 100%;
    box-sizing: border-box;
}

/* Force readable text inside results */
.result-card,
.result-card p,
.result-card li,
.result-card div,
.result-card span,
.result-card strong,
.result-card ul,
.result-card ol {
    color: #2b1205 !important;
}

.result-card h2 {
    color: #ea580c !important;
    font-size: 24px;
    margin-bottom: 10px;
    font-weight: 900;
}

.result-card h3 {
    color: #c2410c !important;
    margin-top: 22px;
    margin-bottom: 10px;
    font-weight: 900;
}

.food-name {
    font-size: 34px;
    font-weight: 900;
    color: #111827 !important;
    margin-bottom: 18px;
    line-height: 1.15;
}

.confidence-box {
    background: linear-gradient(90deg, #fb923c, #f97316);
    color: white !important;
    border-radius: 18px;
    padding: 16px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 18px 0;
    box-shadow: 0 10px 20px rgba(249, 115, 22, 0.25);
}

.confidence-box span,
.confidence-box strong {
    color: #ffffff !important;
}

.confidence-box span {
    font-size: 16px;
    font-weight: 700;
}

.confidence-box strong {
    font-size: 26px;
    font-weight: 900;
}

.result-card ul,
.result-card ol {
    padding-left: 22px;
    line-height: 1.9;
    font-size: 16px;
}

.result-card li {
    margin-bottom: 6px;
}

.error-card {
    background: #fee2e2 !important;
    border: 1px solid #fca5a5;
}

button.primary {
    background: linear-gradient(90deg, #f97316, #ef4444) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 18px rgba(239, 68, 68, 0.25) !important;
}

button.secondary {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

.image-container,
.input-image,
.output-markdown {
    border-radius: 20px !important;
}

label {
    color: #7c2d12 !important;
    font-weight: 700 !important;
}

footer {
    display: none !important;
}

/* Mobile fixes */
@media screen and (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    #snapfuel-header {
        padding: 18px 10px 8px 10px !important;
    }

    #snapfuel-header h1 {
        font-size: 38px !important;
        line-height: 1.05 !important;
    }

    #snapfuel-header p {
        font-size: 14px !important;
        line-height: 1.4 !important;
        color: #7c2d12 !important;
        padding: 0 8px;
    }

    #main-card {
        padding: 14px !important;
        border-radius: 22px !important;
        background: rgba(255, 255, 255, 0.96) !important;
    }

    .result-card {
        padding: 18px !important;
        border-radius: 20px !important;
        background: #ffffff !important;
        color: #2b1205 !important;
        border: 1px solid #fdba74 !important;
        box-shadow: 0 8px 20px rgba(249, 115, 22, 0.14) !important;
    }

    .result-card,
    .result-card p,
    .result-card li,
    .result-card div,
    .result-card span,
    .result-card strong,
    .result-card ul,
    .result-card ol {
        color: #2b1205 !important;
    }

    .result-card h2 {
        font-size: 21px !important;
        color: #ea580c !important;
        line-height: 1.25 !important;
    }

    .result-card h3 {
        font-size: 18px !important;
        color: #c2410c !important;
        line-height: 1.3 !important;
    }

    .food-name {
        font-size: 30px !important;
        line-height: 1.15 !important;
        color: #111827 !important;
        margin-bottom: 16px !important;
    }

    .confidence-box {
        padding: 16px !important;
        border-radius: 18px !important;
        margin: 18px 0 !important;
        display: flex !important;
        gap: 8px !important;
    }

    .confidence-box span,
    .confidence-box strong {
        color: #ffffff !important;
    }

    .confidence-box span {
        font-size: 15px !important;
    }

    .confidence-box strong {
        font-size: 26px !important;
    }

    .result-card ul,
    .result-card ol {
        padding-left: 20px !important;
        line-height: 1.75 !important;
        font-size: 15px !important;
    }

    .result-card li {
        margin-bottom: 8px !important;
    }

    button.primary,
    button.secondary {
        min-height: 46px !important;
        font-size: 16px !important;
        border-radius: 14px !important;
    }
}
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="SnapFuel") as demo:
    gr.HTML("<script>document.title = 'SnapFuel';</script>") # Add this line
    gr.HTML(
        """
        <div id="snapfuel-header">
            <h1>SnapFuel</h1>
            <p>Snap your food. Fuel your goals. Get instant food predictions and calorie estimates.</p>
        </div>
        """
    )

    with gr.Group(elem_id="main-card"):
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="filepath",
                    label="Upload a food image"
                )

                amount_input = gr.Number(
                    label="How much do you have? (grams)",
                    value=None,
                    precision=0,
                    minimum=0,
                    info="Enter the amount in grams to calculate your total calories"
                )

                with gr.Row():
                    clear_button = gr.Button("Clear", variant="secondary")
                    submit_button = gr.Button("Analyze Food", variant="primary")

            with gr.Column(scale=1):
                output_result = gr.HTML(
                    label="Prediction Result",
                    value="""
                    <div class="result-card">
                        <h2>Ready to analyze</h2>
                        <p>Upload a food image and click <strong>Analyze Food</strong>.</p>
                    </div>
                    """
                )

    submit_button.click(
        fn=gradio_predict_wrapper,
        inputs=[input_image, amount_input],
        outputs=output_result
    )

    clear_button.click(
        fn=lambda: (
            None,
            None,
            """
            <div class="result-card">
                <h2>Ready to analyze</h2>
                <p>Upload a food image and click <strong>Analyze Food</strong>.</p>
            </div>
            """
        ),
        inputs=None,
        outputs=[input_image, amount_input, output_result]
    )

demo.launch()
