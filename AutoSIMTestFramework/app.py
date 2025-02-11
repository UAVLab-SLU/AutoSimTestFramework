from flask import Flask, request, jsonify,send_file
from Agents import Agents
from io import BytesIO
import base64

app = Flask(__name__)





@app.route('/generate_scenario', methods=['POST'])
def generate_scenario():
    user_input = request.json.get('user_input', '')
    mission_type = request.json.get('mission_type', 'px4')
    if not user_input:
        return jsonify({"error": "User input is required."}), 400
    scenario_response, context, mission_json_list, environment_json_list = Agents.main(user_input,mission_type)
    
    #
    file1_path = f"user_questions/{user_input}_1.csv"

    return jsonify({
        "scenario_response": scenario_response,
        "context": context,
        "mission_details": mission_json_list,
        "environment_details": environment_json_list,
        "file1": file1_path
    })

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return str(e), 404


@app.route('/analyze', methods=['POST'])
def analyze():
    analytics_input = request.json.get('analytics_input', '')
    if not analytics_input:
        return jsonify({"error": "Analytics input is required."}), 400
    text, images,Analysis = Agents.Analytics_three(analytics_input)

    image_data = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append(img_str)
    
    return jsonify({
        "analysis_result1": text,
        "analysis_result2": image_data,
        "analysis_report" : Analysis,
    })

@app.route('/newanalytics', methods=['POST'])
def newanalytics():
    analytics_input = request.json.get('analytics_input', '')
    images ,response = Agents.new_analytics(analytics_input)
    print(images)
    image_data = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append(img_str)

    return jsonify({
        "output_one": image_data,
        "output_two": response
    })

@app.route('/deepdiverequest', methods=['POST'])
def deepdiverequest():
    deep_dive_input1 = request.json.get('deep_dive_input1', '')
    deep_dive_input2 = request.json.get('deep_dive_input2', '')

    response = Agents.clarification(deep_dive_input1,deep_dive_input2)
    return jsonify({
        "output_one": response,
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
