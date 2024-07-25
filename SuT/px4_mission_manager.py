from pip._vendor import requests

from llm_relay.px4.px4_points_mission import PX4PointsMission


def create_spiral(center, width, height):
    """
    Create points that go like a square spiral from the center, height above the ground.

    :param center: tuple (x_center, y_center, z_center)
    :param width: float, width of the area to cover
    :param height: float, height above the ground
    :return: list of lists, where each sublist is [x, y, z]
    """
    x_center, y_center, z_center = center
    points = []
    x, y = x_center, y_center
    step = width / 20  # Determines the step size for the spiral
    num_steps = int(width / step)

    dx, dy = step, 0  # Initial movement direction (right)
    segment_length = 1  # Initial length of the segment
    segments_per_side = 2  # Number of segments per side before increasing the segment length

    for _ in range(num_steps):
        for _ in range(segment_length):
            points.append([x, y, height + z_center])
            x += dx
            y += dy
        dx, dy = -dy, dx  # Change direction (90 degrees rotation)
        segments_per_side -= 1
        if segments_per_side == 0:
            segment_length += 1
            segments_per_side = 2

    return points



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

        or for automated missions
        {
            "Mission": {
                "name": "Search_and_Rescue_Mission",
                "mode": "auto",
                "center": [0, 0, 0],
                "radius": 10,
                "height": 5
            }
        }
        """
        
        # check type
        if isinstance(mission_raw_json, list):
            print("multiple missions received, using the first one")
            mission_raw_json = mission_raw_json[0]
        else:
            print("single mission received")
            mission_raw_json = mission_raw_json

        if "mode" not in mission_raw_json["Mission"]:
            points = mission_raw_json["Mission"]["param"][1]
        else:
            if mission_raw_json["Mission"]["mode"] == "auto":
                points = create_spiral(mission_raw_json["Mission"]["center"],
                                       mission_raw_json["Mission"]["radius"],
                                       mission_raw_json["Mission"]["height"])
            else:
                raise ValueError("Invalid mode")
        
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
        

