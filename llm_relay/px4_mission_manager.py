from pip._vendor import requests

from llm_relay.px4.px4_points_mission import PX4PointsMission

class PX4MissionManager:
    def __init__(self):
        self.mission = None
        self.mission_ready = False
        self.environment_ready = False

    def set_mission(self, mission_raw_json):
        """
        Set the mission data
        :param mission_raw_json: mission data in raw json format
        [
        {
            "Mission": {
                "name": "Search_and_Rescue_Mission",
                "param": [
                    20,
                    [
                        [
                            0,
                            0,
                            0
                        ],
                        [
                            0,
                            10,
                            -5
                        ],
                        [
                            10,
                            10,
                            -5
                        ],
                        [
                            10,
                            0,
                            -5
                        ],
                        [
                            0,
                            10,
                            -5
                        ],
                        [
                            0,
                            0,
                            0
                        ]
                    ]
                ]
            }
        }
        ]
        """
        
        # check type
        if isinstance(mission_raw_json, list):
            print("multiple missions received, using the first one")
            mission_raw_json = mission_raw_json[0]
        else:
            print("single mission received")
            mission_raw_json = mission_raw_json
            
        points = mission_raw_json["Mission"]["param"][1]
        
        if not isinstance(points, list):
            raise ValueError("points must be a list")
    
        # check each point
        for point in points:
            if not isinstance(point, list):
                raise ValueError("point must be a list")
            if len(point) != 3:
                raise ValueError("point must have 3 elements")
            for element in point:
                if not isinstance(element, (int, float)):
                    raise ValueError("element must be a number")
                
        # speed
        speed = mission_raw_json["Mission"]["param"][0]
        if not isinstance(speed, (int, float)):
            raise ValueError("speed must be a number")
            
        
        self.mission = PX4PointsMission(points, speed)
        self.mission_ready = True
        
        if self.environment_ready:
            self.start_mission()

    def set_environment(self):
        """
        Set the environment data
        """
        self.environment_ready = True

        if self.mission_ready:
            self.start_mission()

    def start_mission(self):
        """
        Start the mission
        :return: 
        """
        if not self.mission_ready:
            raise ValueError("Mission is not ready")
        if not self.environment_ready:
            raise ValueError("Environment is not ready")
        
        self.mission.start()
        self.notify_done()
        self.reset_state()
        
    def reset_state(self):
        self.mission_ready = False
        self.environment_ready = False

    def notify_done(self):
        """
        Notify the client that the mission is done
        :return:
        """
        # get host ip
        requests.post("http://192.168.1.181:5000/llm_mission_done_notify")
        

