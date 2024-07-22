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
        analysis_report = data.get('analysis_report', 'No Report Found')
        images = [Image.open(BytesIO(base64.b64decode(img_str))) for img_str in images_base64]
        return analysis_result1, images, analysis_report
    else:
        return "Failed to get response from API", [], "No Report Found"

def deep_dive_request(deep_dive_input1,deep_dive_input2):
    url = "http://localhost:5000/deepdiverequest"
    response = requests.post(url, json={"deep_dive_input1": deep_dive_input1,"deep_dive_input2":deep_dive_input2})
    
    if response.status_code == 200:
        data = response.json()
        analysis_result1 = data.get('output_one', 'No analysis result found')
        # images_base64 = data.get('analysis_result2', [])
        # analysis_report = data.get('analysis_report', 'No Report Found')
        # images = [Image.open(BytesIO(base64.b64decode(img_str))) for img_str in images_base64]
        return analysis_result1
    else:
        return "Failed to response to API"
    # return f"Deep Dive analysis for input: {deep_dive_input}"

def more_analytics_request(more_analytics_input):
    url = "http://localhost:5000/newanalytics"
    response = requests.post(url, json={"analytics_input": more_analytics_input})

    if response.status_code == 200:
        data = response.json()
        images_base64 = data.get('output_one', [])
        analysis_report = data.get('output_two', 'No Report Found')
        images = [Image.open(BytesIO(base64.b64decode(img_str))) for img_str in images_base64]
        return images, analysis_report
    else:
        return [], "Failed to get response from API"

with gr.Blocks() as demo:
    # Main Analytics Section
    with gr.TabItem(label="sUAS Analytics"):
        gr.Markdown("# sUAS Analytics")
        analytics_input = gr.Textbox(lines=2, placeholder="Enter your analytics input here...")
        analytics_button = gr.Button("Submit")
        analysis_result = gr.Textbox(label="Analysis Result")
        analysis_images = gr.Gallery(label="Analysis Images")
        analysis_report = gr.Textbox(label="Analysis Report")
        analytics_button.click(send_analytics_request, inputs=analytics_input, outputs=[analysis_result, analysis_images, analysis_report])
    
    # Deep Dive Section
    with gr.TabItem(label="Deep Dive"):
        gr.Markdown("# Deep Dive")
        deep_dive_input1 = gr.Textbox(lines=2, placeholder="Enter your deep dive input here...")
        deep_dive_input2 = gr.Textbox(lines=2, placeholder="Ask your question here")
        deep_dive_button = gr.Button("Submit")
        deep_dive_result = gr.Textbox(label="Deep Dive Result")
        deep_dive_button.click(deep_dive_request, inputs=[deep_dive_input1,deep_dive_input2],outputs=[deep_dive_result])
    
    # More Analytics Section
    with gr.TabItem(label="More Analytics"):
        gr.Markdown("# More Analytics")
        more_analytics_input = gr.Textbox(lines=2, placeholder="Enter your more analytics input here...")
        more_analytics_button = gr.Button("Submit")
        more_analytics_images = gr.Gallery(label="Analysis Images")
        more_analytics_result = gr.Textbox(label="More Analytics Result")
        more_analytics_button.click(more_analytics_request, inputs=more_analytics_input, outputs=[more_analytics_images, more_analytics_result])

demo.launch(share=True, debug=True)











