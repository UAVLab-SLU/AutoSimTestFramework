import gradio as gr
import requests

def send_to_flask_api(user_input, mission_type):
    # URL of your Flask API
    url = "http://localhost:5000/generate_scenario"
    
    # Send POST request to the Flask API
    response = requests.post(url, json={"user_input": user_input, "mission_type": mission_type})
    
    if response.status_code == 200:
        # Extract data from response
        data = response.json()
        scenario_response = data.get('scenario_response', 'No response found')
        context = data.get('context', 'No context provided')
        mission_details = data.get('mission_details', 'No mission details provided')
        environment_details = data.get('environment_details', 'No environment details provided')
        
        file1 = data.get("file1", "error to download")
        download_link1 = f'<a href="http://localhost:5000/download/{file1}" download="{file1}">Download CSV 1</a>'
        
        return scenario_response, context, str(mission_details), str(environment_details), download_link1
    else:
        return "Failed to get response from API", "", "", "", ""

def validate_json(json_input, json_type):
    # URL of your Flask API for JSON validation
    url = "http://localhost:5000/validate_json"
    
    # Send POST request to the Flask API
    response = requests.post(url, json={"json_input": json_input, "json_type": json_type})
    
    if response.status_code == 200:
        # Extract validation result from response
        data = response.json()
        validation_result = data.get('validation_result', 'No validation result found')
        return validation_result
    else:
        return "Failed to get response from API"

# Define the Gradio interface with Blocks
with gr.Blocks() as demo:
    gr.Markdown("# sUAS Scenario Generator")
    gr.Markdown("Provide your high-level requirements, and our AI agents will handle the rest.")

    with gr.Tab("Generate Scenario"):
        gr.Markdown("### Generate Mission Scenario")
        
        user_input = gr.Textbox(lines=2, placeholder="Enter your user input here...", label="User Input")
        mission_type = gr.Dropdown(choices=['px4', 'drone_response'], label="Select Mission Type")
        generate_button = gr.Button("Generate Scenario")
        
        scenario_response = gr.Textbox(label="Scenario Response")
        context = gr.Textbox(label="Context")
        mission_details = gr.Textbox(label="Mission Details")
        environment_details = gr.Textbox(label="Environment Details")
        download_link1 = gr.HTML(label="Download CSV 1")
        
        generate_button.click(
            send_to_flask_api,
            inputs=[user_input, mission_type],
            outputs=[scenario_response, context, mission_details, environment_details, download_link1]
        )

    with gr.Tab("Validate JSON"):
        gr.Markdown("### Validate Generated JSON")
        
        json_input = gr.Textbox(lines=10, placeholder="Enter the JSON here...", label="JSON Input")
        json_type = gr.Dropdown(choices=['px4', 'drone_response','environment_json'], label="Select JSON Type")
        validate_button = gr.Button("Validate JSON")
        
        validation_result = gr.Textbox(label="Validation Result")
        
        validate_button.click(
            validate_json,
            inputs=[json_input, json_type],
            outputs=validation_result
        )

# Launch the interface
demo.launch(share=True, debug=True)

