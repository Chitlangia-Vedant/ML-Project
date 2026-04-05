import gradio as gr
import pickle

# Load model (Using relative path for easy sharing/deployment)
# Note: Ensure this pickle file contains a full Pipeline (Vectorizer + Model)
try:
    model = pickle.load(open("G:/Code/ML Project/model.pkl", "rb"))
except FileNotFoundError:
    print("Warning: model.pkl not found. Make sure the path is correct.")

def predict(text):
    if not text.strip():
        return {"Please enter text": 1.0}

    # Standard hard prediction (What you currently have)
    pred = model.predict([text])[0]
    
       # Returning your original dictionary logic, mapped nicely for Gradio
    if pred == 1:
        return {"Positive": 1.0, "Negative": 0.0}
    else:
        return {"Negative": 1.0, "Positive": 0.0}

# UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 IMDB Review Sentiment Analysis App")
    gr.Markdown("Analyze movie review sentiment instantly!")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                placeholder="Type your movie review here...",
                lines=5,
                label="Your Text"
            )
            
            # Adding examples makes testing much faster for users
            gr.Examples(
                examples=[
                    "This movie was an absolute masterpiece! The acting was phenomenal.",
                    "Terrible writing, boring plot, and a complete waste of two hours.",
                    "It was okay. Not the best I've seen, but fine for a rainy Sunday."
                ],
                inputs=text_input
            )

        with gr.Column():
            output = gr.Label(label="Prediction Confidence")

    with gr.Row():
        btn = gr.Button("🚀 Analyze", variant="primary")
        # Using Gradio's built in ClearButton to cleanly wipe specific components
        clear = gr.ClearButton(components=[text_input, output], value="🗑️ Clear")

    btn.click(fn=predict, inputs=text_input, outputs=output)

demo.launch()
