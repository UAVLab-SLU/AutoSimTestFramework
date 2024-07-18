import gradio as gr
import requests
from PIL import Image
from io import BytesIO
import base64


def send_analytics_request(analytics_input):
    url = "http://localhost:5000/analyze"
    response = requests.post(url, json={"analytics_input": analytics_input})
    
    if response.status_code == 200:
        data = response.json()
        analysis_result1 = data.get('analysis_result1', 'No analysis result found')
        images_base64 = data.get('analysis_result2', [])
        
        images = [Image.open(BytesIO(base64.b64decode(img_str))) for img_str in images_base64]
        return analysis_result1, images
    else:
        return "Failed to get response from API", []



analytics_interface = gr.Interface(
    fn=send_analytics_request,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your analytics input here...")],
    outputs=[
        gr.Textbox(label="Analysis Result"),
        gr.Gallery(label="Analysis Images")
    ],
    title="sUAS Analytics",
    description="Enter your analytics input and get the analysis results."
)

analytics_interface.launch(share=True, debug=True)

