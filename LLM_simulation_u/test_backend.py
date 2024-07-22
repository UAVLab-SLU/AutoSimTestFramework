from flask import Flask, request, jsonify, send_file
import pandas as pd
import os

app = Flask(__name__)

@app.route('/test_csv', methods=['POST'])
def test_csv():
    file_name = request.json.get('file_name', '')
    if not file_name:
        return jsonify({"error": "File name is required."}), 400
    
    file_path = os.path.join("user_questions", file_name)  # Ensure the file path is correct
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found."}), 404
        
        return jsonify({
            "download_link": f"http://localhost:5000/download/{file_name}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join("user_questions", filename)
    try:
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return str(e), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
