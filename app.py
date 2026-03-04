import gradio as gr
import pandas as pd
from address_quality_detector import process_and_predict

def check_address(address_line1, address_line2, landmark, city, state):

    # Validate BEFORE creating DataFrame
    if not any([address_line1, address_line2, landmark, city, state]):
        return "⚠️ Please enter at least one field."

    # Create DataFrame
    new_data = pd.DataFrame([{
        "address": "",   # IMPORTANT: your model expects this column
        "address_line1": address_line1,
        "landmark": landmark,
        "city": city,
        "state": state
    }])

    # Call model
    result_df = process_and_predict(new_data)

    # Extract values from returned DataFrame
    label = result_df.loc[0, "predicted_quality"]
    prob = result_df.loc[0, "probability"]

    confidence = round(prob * 100, 2)

    if label.lower() == "valid":
        return f"✅ VALID ADDRESS\nConfidence: {confidence}%"
    elif label.lower() == "partially_valid":
        return f"⚠️ PARTIALLY VALID\nConfidence: {confidence}%"
    elif label.lower() == "partially_invalid":
        return f"⚠️ PARTIALLY INVALID\nConfidence: {confidence}%"
    else:
        return f"❌ INVALID ADDRESS\nConfidence: {confidence}%"

iface = gr.Interface(
    fn=check_address,
    inputs=[
        gr.Textbox(label="Address Line 1"),
        gr.Textbox(label="Address Line 2"),
        gr.Textbox(label="Landmark"),
        gr.Textbox(label="City"),
        gr.Textbox(label="State"),
    ],
    outputs="text",
    title="🏠 Address Quality Detector",
    description="Enter at least one field."
)

if __name__ == "__main__":
    iface.launch()
# iface.launch(share=False, inbrowser=True)
