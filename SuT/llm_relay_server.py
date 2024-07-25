import logging

from flask import Flask, request
from flask_cors import CORS

from llm_relay.px4_mission_manager import PX4MissionManager

print(
"""
 ____      _             
|  _ \ ___| | __ _ _   _ 
| |_) / _ \ |/ _` | | | |
|  _ <  __/ | (_| | |_| |
|_| \_\___|_|\__,_|\__, |
                   |___/ 
"""
)

app = Flask(__name__)
log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
CORS(app)

mission_manager = PX4MissionManager()


@app.route('/')
def hello():
    return 'Hello, PX4 Relay Server!'


@app.route('/mission', methods=['POST'])
def mission():
    """
    Receive mission data from the client
    :returns: mission data
    """
    json_data = request.get_json()
    mission_manager.set_mission(json_data)
    return 'Mission received!'

@app.route('/environment_done_notify', methods=['POST'])
def environment():
    """
    Receive environment done notification from the client
    :returns: environment done notification
    """
    mission_manager.set_environment()
    return 'Environment done notification received!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)