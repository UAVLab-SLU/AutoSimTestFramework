import gradio as gr
import requests

def send_to_flask_api(user_input,mission_type):
    # URL of your Flask API
    url = "http://localhost:5000/generate_scenario"
    
    # Send POST request to the Flask API
    response = requests.post(url, json={"user_input": user_input,"mission_type":mission_type})
    
    if response.status_code == 200:
        # Extract data from response
        data = response.json()
        scenario_response = data.get('scenario_response', 'No response found')
        context = data.get('context', 'No context provided')
        mission_details = data.get('mission_details', 'No mission details provided')
        environment_details = data.get('environment_details', 'No environment details provided')
        
        file1 = data.get("file1","error to download")
        # download_link1 = f"http://localhost:5000/download/{file1}"
        download_link1 = f'<a href="http://localhost:5000/download/{file1}" download="{file1}">Download CSV 1</a>'

        
        return scenario_response, context, str(mission_details), str(environment_details),download_link1
    else:
        return "Failed to get response from API", "", "", ""

def send_analytics_request(analytics_input):
    # URL of your Flask API for analytics
    url = "http://localhost:5000/analyze"
    
    # Send POST request to the Flask API
    response = requests.post(url, json={"analytics_input": analytics_input})
    
    if response.status_code == 200:
        # Extract data from response
        data = response.json()
        analysis_result1 = data.get('analysis_result1', 'No analysis result 1 found')
        images_base64 = data.get('analysis_result2', [])
        
        # Decode base64 images
        images = []
        for img_str in images_base64:
            image = Image.open(BytesIO(base64.b64decode(img_str)))
            images.append(image)
        
        return analysis_result1, images
    else:
        return "Failed to get response from API", [], []


# Define the Gradio interface
iface = gr.Interface(
    fn=send_to_flask_api,
    inputs=[
    gr.Textbox(lines=2, placeholder="Enter your user input here..."),
    gr.Dropdown(choices=['px4', 'drone_response'], label="Select Mission Type")],
    outputs=[
        gr.Textbox(label="Scenario Response"),
        gr.Textbox(label="Context"),
        gr.Textbox(label="Mission Details"),
        gr.Textbox(label="Environment Details"),
        #
        gr.HTML(label="Download CSV 1"),
    ],
    title="sUAS Scenario Generator",
    description="Give your High level requirement and our Ai Agents will take care of the Rest"
)




iface.launch(share=True,debug=True)
