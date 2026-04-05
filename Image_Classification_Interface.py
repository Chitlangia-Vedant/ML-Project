import gradio as gr
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model("./vgg_model.keras") 

labels = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# Prediction function
def predict(img):
    # 1. Ensure the image is in RGB format to drop any alpha channels (transparency)
    img = img.convert("RGB") 
    
    # 2. Convert to numpy and resize
    img = np.array(img)
    img = tf.image.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # 3. Predict
    pred = model.predict(img)[0]

    # 4. Create DataFrame for bar chart
    df = pd.DataFrame({
        "Class": labels,
        "Confidence": pred.tolist() # Convert to standard Python floats for safety
    })

    return df.sort_values(by="Confidence", ascending=False)

# Wrapper
def predict_with_status(img):
    if img is None:
        return None, None, "⚠️ Please upload an image."
    
    df = predict(img)
    top3 = df.head(3)

    # Convert top3 to label format expected by gr.Label
    label_output = {row["Class"]: float(row["Confidence"]) for _, row in top3.iterrows()}

    return label_output, df, "✅ Prediction complete!"

# UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧠 CIFAR-10 Classifier with Visualization")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        label_output = gr.Label(num_top_classes=3)

    btn = gr.Button("🚀 Predict", variant="primary")

    bar_plot = gr.BarPlot(
        x="Class",
        y="Confidence",
        title="Prediction Confidence for All Classes",
        tooltip=["Class", "Confidence"], # Adds nice hover effects
        y_lim=[0, 1] # Locks the Y-axis to 0-1 (0% to 100%) for consistency
    )

    status = gr.Markdown()

    btn.click(
        fn=predict_with_status,
        inputs=image_input,
        outputs=[label_output, bar_plot, status]
    )

demo.launch()