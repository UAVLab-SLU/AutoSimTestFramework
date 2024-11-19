import json

def validate_px4_mission(mission_json):
    issues = []
    
    # Rule: Check mission name and parameters structure
    if "Mission" not in mission_json or "name" not in mission_json["Mission"] or "param" not in mission_json["Mission"]:
        issues.append("Mission JSON structure is missing required fields.")
    
    # Rule: Check if mission name is a string
    if "Mission" in mission_json and not isinstance(mission_json["Mission"].get("name"), str):
        issues.append("Mission name should be a string.")
    
    # Rule: Validate velocity and waypoints in param
    params = mission_json["Mission"].get("param", [])
    if not isinstance(params, list) or len(params) < 2:
        issues.append("Parameters should be a list with velocity and waypoints.")
    else:
        velocity = params[0]
        waypoints = params[1]

        if not (isinstance(velocity, int) and 1 <= velocity <= 100):
            issues.append("Velocity should be an integer between 1 and 100.")

        if not (isinstance(waypoints, list) and len(waypoints) >= 5):
            issues.append("There should be at least five waypoints.")

        for waypoint in waypoints:
            if not (isinstance(waypoint, list) and len(waypoint) == 3 and all(isinstance(coord, (int, float)) for coord in waypoint)):
                issues.append("Each waypoint should be a list of three numerical coordinates.")
                
    return issues if issues else ["PX4 Mission JSON is valid."]


def validate_drone_response_mission(mission_json):
    issues = []
    
    # Rule: Check if states are present
    if "states" not in mission_json or not isinstance(mission_json["states"], list):
        issues.append("Drone response JSON structure is missing 'states' list.")
    else:
        state_names = {"Takeoff", "BriarWaypoint", "BriarHover", "LandLocation", "Land", "Disarm"}
        
        for state in mission_json["states"]:
            if "name" not in state or state["name"] not in state_names:
                issues.append(f"State {state} has an invalid or missing name.")
            
            # Check specific attributes per state
            if state["name"] == "Takeoff":
                if not all(attr in state["args"] for attr in ["altitude", "speed", "altitude_threshold"]):
                    issues.append("Takeoff state missing required attributes.")
            
            elif state["name"] == "BriarWaypoint":
                waypoint = state["args"].get("waypoint", {})
                if not all(key in waypoint for key in ("latitude", "longitude", "altitude")):
                    issues.append("BriarWaypoint missing required waypoint attributes.")
            
            elif state["name"] == "BriarHover" and "hover_time" not in state["args"]:
                issues.append("BriarHover state missing 'hover_time'.")

    return issues if issues else ["Drone response mission JSON is valid."]


def validate_environment_specification(environment_json):
    issues = []
    
    # Rule: Check environment structure and required keys
    if "environment" not in environment_json or "monitors" not in environment_json:
        issues.append("Environment JSON is missing required 'environment' or 'monitors' sections.")
    else:
        # Check environment values
        environment = environment_json["environment"]
        if not isinstance(environment.get("Wind", {}).get("Direction"), str):
            issues.append("Wind Direction should be a string (N, NE, E, SE, S, SW, W, NW).")
        
        if not isinstance(environment.get("Wind", {}).get("Velocity"), int):
            issues.append("Wind Velocity should be an integer.")
        
        origin = environment.get("Origin", {})
        if not all(key in origin for key in ("Latitude", "Longitude", "Height", "Name")):
            issues.append("Origin is missing one of 'Latitude', 'Longitude', 'Height', or 'Name'.")
        
        # Check monitor parameters
        monitors = environment_json["monitors"]
        required_monitors = {
            "circular_deviation_monitor": [15],
            "collision_monitor": [],
            "point_deviation_monitor": [15],
            "min_sep_dist_monitor": [10, 1],
            "landspace_monitor": [[0, 0]],
            "no_fly_zone_monitor": [[[0, 0], [0, 0], [0, 0]]],
            "wind_monitor": [0.5]
        }
        
        for monitor, params in required_monitors.items():
            if monitor not in monitors:
                issues.append(f"{monitor} is missing in monitors.")
            elif not isinstance(monitors[monitor].get("param"), list):
                issues.append(f"{monitor} param should be a list.")
    
    return issues if issues else ["Environment specification JSON is valid."]


# Example JSON for validation
#px4_json =  


print(validate_px4_mission(px4_json))
# print(validate_drone_response_mission(drone_response_json))
print(validate_environment_specification(environment_json))
