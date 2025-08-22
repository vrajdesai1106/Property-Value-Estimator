import gradio as gr
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Prediction function
def predict_price(bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                  view, condition, sqft_above, sqft_basement, yr_built, yr_renovated):
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated
    }])
    return f"${model.predict(input_df)[0]:,.2f}"

# Gradio Blocks with styling
with gr.Blocks(css="""
    body {background-color: #f0f8ff;}
    #predict_btn {background-color: #1E90FF; color: white;}
""") as demo:

    # Page header
    gr.Markdown("<h1 style='text-align:center; color:#1E90FF;'>AI Property Price Estimator</h1>")
    gr.Markdown("<p style='text-align:center; color:#555;'>Estimate property prices instantly</p>")

    # Input layout
    with gr.Row():
        with gr.Column():
            bedrooms = gr.Slider(1, 10, step=1, label="Bedrooms")
            bathrooms = gr.Slider(1, 10, step=0.5, label="Bathrooms")
            floors = gr.Slider(1, 5, step=1, label="Floors")
            waterfront = gr.Dropdown([0,1], label="Waterfront")
            view = gr.Slider(0,4,1, label="View")
            condition = gr.Slider(1,5,1, label="Condition")
        with gr.Column():
            sqft_living = gr.Number(label="Sqft Living", value=0)
            sqft_lot = gr.Number(label="Sqft Lot", value=0)
            sqft_above = gr.Number(label="Sqft Above", value=0)
            sqft_basement = gr.Number(label="Sqft Basement", value=0)
            yr_built = gr.Number(label="Year Built", value=1800)
            yr_renovated = gr.Number(label="Year Renovated", value=0)

    # Predict button and output
    predict_btn = gr.Button("Predict Price", elem_id="predict_btn")
    output = gr.Textbox(label="Predicted Price")

    predict_btn.click(fn=predict_price, 
                      inputs=[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
                              view, condition, sqft_above, sqft_basement, yr_built, yr_renovated],
                      outputs=output)

# Launch app
demo.launch(share=True)
