#imports
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss
import re
from PIL import Image 
import requests 
import torch
import matplotlib.pyplot as plt
import json
from transformers import AutoModelForCausalLM ,AutoTokenizer,AutoProcessor,pipeline
#load_data
from transformers import BitsAndBytesConfig

import argparse
data = pd.read_csv("knowledga_base.csv")

Analytics_data = pd.read_csv("Analytics_knowledge.csv")

with open('missions.json', 'r') as file:
    mission_json_data = json.load(file)


with open('/home/uav/Documents/AI_Hunter/LLMS/columns.txt', 'r') as file:
    analytics_columns = file.read()

#load knowledge base
knowledge = data.drop(["filename","synopsis"],axis=1)
knowledge_tensor = torch.tensor(knowledge.values, dtype=torch.float32)
normalized_embeddings  = F.normalize(knowledge_tensor, p=2.0, dim=1)
dimension = knowledge_tensor.shape[1]
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res,dimension)
index.add(normalized_embeddings.numpy())

##load the Analytics
knowledge_a = Analytics_data.drop(["columns","Unnamed: 0"],axis=1)
knowledge_tensor_a = torch.tensor(knowledge_a.values, dtype=torch.float32)
normalized_embeddings_a  = F.normalize(knowledge_tensor_a, p=2.0, dim=1)
dimension_a = knowledge_tensor_a.shape[1]
res_a = faiss.StandardGpuResources()
index_a = faiss.GpuIndexFlatIP(res_a,dimension_a)
index_a.add(normalized_embeddings_a.numpy())

#embeddings
def get_embeddings(query):
    embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = embed_model.encode(query)
    torch_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    norm_embeddings  = F.normalize(torch_embeddings, p=2.0, dim=-1)
    return norm_embeddings

def search(query,k):
    query_embeddings = get_embeddings(query)
    query_embeddings = query_embeddings.unsqueeze(0)
    cos_sim,cos_indices = index.search(query_embeddings.numpy(),k)
    return cos_sim,cos_indices

def search_a(query,k):
    query_embeddings = get_embeddings(query)
    query_embeddings = query_embeddings.unsqueeze(0)
    cos_sim,cos_indices = index_a.search(query_embeddings.numpy(),k)
    return cos_sim,cos_indices
    
def combine_context(contentlist):
    return "".join(contentlist)



class Agents:

    model = None
    tokenizer = None

    # @classmethod
    # def load_model(cls):
    #     if cls.model is None or cls.tokenizer is None:
    #         model_id = "microsoft/Phi-3-medium-128k-instruct"
    #         quantization_config = BitsAndBytesConfig(load_in_4bit=True,llm_int4_threshold=6.0,llm_int8_skip_modules=None,trust_remote_code=True)
    #         cls.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", quantization_config=quantization_config,trust_remote_code=True)#load_in_4bit=True, trust_remote_code=True, torch_dtype="auto")
    #         cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     return cls.model, cls.tokenizer


    @staticmethod
    def model_x(messages, temp, sample,tokens):
        # model_id = "microsoft/Phi-3-medium-128k-instruct"
        # llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", load_in_4bit=True, trust_remote_code=True, torch_dtype="auto")
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        model, tokenizer = Agents.load_model()
        generation_args = { 
            "max_new_tokens": tokens, 
            "temperature": temp, 
            "do_sample": sample, 
        } 

        pipe = pipeline("text-generation",model=model,tokenizer=tokenizer,)
        output = pipe(messages, **generation_args)
        return output[0]['generated_text']

    @staticmethod
    def scenario_generator_Agent(user_input):
        cosine_similarity, indices = search(user_input, 5)
        contextlist = data.iloc[indices[0]]["synopsis"]
        context = combine_context(contextlist)
        text = f"""
          I want to design scenarios for simulation testing of sUAS systems. The scenarios must contain information about the environmental context, the sUAS mission, 
         and the properties of the sUAS that must be tested.
    
        Senarios : #number
        
        Environmental Context:
        
            Location: The mission spans from Chicago downtown to Saint Louis University, characterized by tall buildings, narrow alleys, and significant structural damage.
            Weather: Wind speed of 10 meters per second (m/s), affecting flight stability and sensor performance.
            Temperature: 20°C, ideal for drone operations but potentially challenging for battery efficiency in prolonged missions.
            Lighting Conditions :Transitioning to dusk, requiring enhanced reliance on night vision and thermal imaging sensors.
            GPS Signal Quality :Generally high, but occasional interferences due to tall buildings and underground obstructions.
            Altitude: Varies from ground level (0 meters) to 100 meters, necessitating versatile altitude adjustments to navigate obstacles.
            Obstacles: Power lines and other urban infrastructure elements posing potential collision risks.
            Network/communication quality : Weak and intermittent in areas with dense structural damage and under tall buildings.
        
        Mission :- very detailed explanation of mission 
        
        sUAS Configuration and Properties:
        
            Sensors:GPS: For navigation and location tracking, with occasional signal interference.Thermal Camera: For detecting heat signatures of missing persons, especially useful as visibility decreases.Acoustic Sensors: To detect sounds indicating the presence of missing persons or wildlife, with specified noise levels to filter out ambient sounds.
            Payload Weight: 2 kg, including rescue equipment (e.g., first aid kit, communication device) and sensors.
            Battery Life: 45 minutes per flight, requiring efficient planning for search patterns and recharging logistics.
        
            
        sUAS Properties to be Tested:
        
            1. Maneuverability: 
            2. Durability: 
            3. Stability in Low Light:
            4. Communication Range: 
        
        any thing you might think its appropriate
        use this past drone insidents context for making best senarios so that we can test all senarios make sure we build best test scripts
            {context}   
        Generate five unique senarios
    
        rules to follow strictly:
            1) Every Senarios should be completely different from each other with absolute 0 overlap between inter senarios
            2) intra context of senario should be completely similar
            3) Should cover 100 % all Parameters of :
                1)Environmental Context (all 8 params),2)Mission, 3)sUAS Configuration and Properties,4)sUAS Properties to be Tested)
            4) use the High level Goal described the Enginner  
            
            Described in Above example case & never copy it just use format of it 
        
            sUas test Engineer High Level Goal: {user_input}"""
        
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ] 
        
        temp = 0.0
        sample = False
        tokens = 10000
        response = Agents.model_x(messages, temp, sample,tokens)
        return response[-1]["content"],context

    @staticmethod
    def helper_for_mission_and_environment(response):
        pattern = r'Mission:\s*(.*?)\n\n'
        missions = re.findall(pattern, response)
        # rest = re.findall(r'^(?!Mission: ).*', response, re.MULTILINE)
        scenarios = re.split(r'\n\nScenario \d+:\n\n', response)
        scenarios = [s for s in scenarios if s.strip()]
        pattern_r = r'Mission:\n\n.*?(?=\n\nsUAS Configuration and Properties:|\n\nScenario \d+:|$)'
        cleaned_scenarios = [re.sub(pattern_r, '', scenario, flags=re.DOTALL).strip() for scenario in scenarios]
        return missions, cleaned_scenarios

    @staticmethod
    def Mission_Agent(missions,mission_type):

        if mission_type == "px4":
            json = mission_json_data[mission_type]
            text = f"""
            Mission details: {missions}
            
            Prompt:

            Create a JSON object for a drone mission with the following structure and capabilities. The mission should include a mission name and parameters such as the drone's velocity in meters per second and a list of coordinates the drone will fly to in order. The coordinates should be in Cartesian format (x, y, z), with each coordinate representing a point in space.
            The mission should have at least five waypoints.
            Note: In this coordinate system, +z indicates downward movement and -z indicates upward movement.
            Example JSON Structure:
            {json}
            Instructions:
            1)Generate a mission name that reflects the purpose or nature of the mission.
            2)Drone Velocity :- Randomly select value between 1 and 100 meters per second.
            3)Create a list not less than five waypoints. Each waypoint should be represented by a list of three integers or floating-point numbers (x, y, z), denoting the position in Cartesian coordinates.
            4)Ensure the waypoints form a logical flight path for the drone, such as a square, circle, or a more complex pattern (very important conisder this
            as rule)
            5)Provide diverse values for the coordinates and different geometry (way points)  to demonstrate a variety of possible missions
            rule:-
            1) dont change the Structure of Example json follow it 100% all the time
            """
        if mission_type == "drone_response":
            json = mission_json_data[mission_type]
            text = f"""
            Mission details: {missions}
            prompt :
            Create a JSON object for a drone mission with the following structure and capabilities.
            The mission should include a list of states, each with specific attributes and transitions.
            The JSON should represent a sequence of drone operations, starting from takeoff, navigating through waypoints, hovering, and eventually landing and disarming. Ensure the mission includes:

            States:
                Takeoff:
                Attributes: altitude, speed, and altitude_threshold.
                Transitions: Targets for succeeded and failed takeoff.
                Waypoint Navigation:
                Attributes: waypoint (latitude, longitude, altitude), speed.
                Transitions: Targets for succeeded waypoints.
                Hover:
                Attributes: hover_time.
                Transitions: Targets for succeeded hover.
                Landing:
                Attributes: waypoint (latitude, longitude, altitude), speed.
                Transitions: Targets for succeeded waypoints and landing.
                Disarm:
                Transitions: Targets for succeeded disarm and mission completion.
            Example JSON Structure:
            {json}
            Instructions:
            Generate state names: Reflect the purpose or nature of each state.
            Drone Speed: Randomly select values between 1 and 10 meters per second for each state involving movement.
            Waypoints: Create diverse waypoints with at least two different coordinates to demonstrate various mission patterns.
            Logical Flight Path: Ensure the waypoints form a logical and feasible flight path for the drone."""

        
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of generating content as per your requirement"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 10000
        agent_response = Agents.model_x(messages, temp, sample,tokens)
        return agent_response[-1]["content"]

    @staticmethod
    def Environment_specification_Agent(rest):
        # Environment Specification Generator Agent
        json = """JSON
         "environment": {"Wind": {"Direction": "NE","Velocity": 15},
                         "Origin": {"Latitude": value,"Longitude": value,"Height": 0, "Name": "Specify Region"},
                         "TimeOfDay": "10:00:00","UseGeo": true, "UseCFD": true },
         "monitors": {"circular_deviation_monitor": {"param": [15]},
                      "collision_monitor": {"param": []},
                      "point_deviation_monitor": {"param": [15]},
                      "min_sep_dist_monitor": {"param": [10,1]},
                      "landspace_monitor": {"param": [[[0,0]]]},
                      "no_fly_zone_monitor": { "param": [[[0,0],[0,0],[0,0]]]},
                      "wind_monitor": { "param": [0.5]}}"""
        text = f"""
        Generate a JSON Object for sUAS System Simulation
        Create a JSON object based on the provided context and the example JSON format below. The goal is to generate an environment for a drone simulation system to test various scenarios and build optimal drone scripts.
        Context:
        {rest}
        Example JSON Format:
        {json}
        Field Values:
        Direction: string (N, NE, E, SE, S, SW, W, NW)
        Velocity: int (m/s)
        Height: int (m)
        UseGeo: Boolean (true for lat/long/height, false for cartesian)
        UseCFD: Boolean (true or False ) 
        Monitors:
        circular_deviation_monitor: [int] (meters)
        collision_monitor: [] (no parameters)
        point_deviation_monitor: [int] (meters)
        min_sep_dist_monitor: [int, int] (meters, meters)
        landspace_monitor: [float, float] (latitude, longitude)
        no_fly_zone_monitor: [[float, float], ...] (list of points for a polygon)
        Instructions:
        1)Generate a JSON object based on the provided context, following the example JSON structure 100% & add json so that it will help me to write
        regex and extract the json.
        2)Change the values randomly to create a new environment , do not copy the example JSON.
        3)Ensure no overlap between the example JSON and the generated JSON.
        4)Please generate a new JSON object for the drone simulation environment based on the provided context and rules
        Rules:
        1) dont change the Structure of Example json
        """
                
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 5000
        agent_response = Agents.model_x(messages, temp, sample,tokens)
        return agent_response[-1]["content"]
        
    @staticmethod
    def json_extraction(environment_responses, mission_responses):
        environment_response_list, mission_response_list, indices = [], [], []

        # Regular expression pattern to match JSON
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)

        # Assuming environment_responses and mission_responses are of the same length
        for i in range(len(environment_responses)):
            environment_response = environment_responses[i]
            mission_response = mission_responses[i]
            try:
                # Extract and parse environment JSON
                json_match_env = json_pattern.search(environment_response)
                if json_match_env:
                    json_data_env = json.loads(json_match_env.group(0))
                else:
                    raise ValueError("No JSON found in environment response")
        
                # Extract and parse mission JSON
                json_match_mission = json_pattern.search(mission_response)
                if json_match_mission:
                    json_data_mission = json.loads(json_match_mission.group(0))
                else:
                    raise ValueError("No JSON found in mission response")
        
                # If both extractions are successful, append to lists
                environment_response_list.append(json_data_env)
                mission_response_list.append(json_data_mission)
                indices.append(i)
        
            except Exception as e:
                # If an error occurs in either, skip both
                print(f"Error processing pair at index {i}: {e}")
        
        return environment_response_list, mission_response_list, indices
    # def json_extraction(environment_responses, mission_responses):
    #     environment_response_list, mission_response_list,index = [], [],[]
    
    #     # Assuming environment_responses and mission_responses are of the same length
    #     for i in range(len(environment_responses)):
    #         environment_response = environment_responses[i]
    #         mission_response = mission_responses[i]
    #         try:
    #             # Extract and parse environment JSON
    #             json_match_env = re.search(r'\{.*\}', environment_response, re.DOTALL)
    #             if json_match_env:
    #                 json_data_e = json.loads(json_match_env.group(0))
    #             else:
    #                 raise ValueError("No JSON found in environment response")
    
    #             # Extract and parse mission JSON
    #             json_match_mission = re.search(r'\{.*\}', mission_response, re.DOTALL)
    #             if json_match_mission:
    #                 json_data_m = json.loads(json_match_mission.group(0))
    #             else:
    #                 raise ValueError("No JSON found in mission response")
    
    #             # If both extractions are successful, append to lists
    #             environment_response_list.append(json_data_e)
    #             mission_response_list.append(json_data_m)
    #             index.append(i)
    
    #         except Exception as e:
    #             # If an error occurs in either, skip both
    #             print(f"Error processing pair at index {i}: {e}")
    
    #     return environment_response_list, mission_response_list,index

    # def json_extraction(environment_responses, mission_responses):#wrong
    #     environment_response_list, mission_response_list = [], []
    
    #     # Process each environment response
    #     for environment_response in environment_responses:
    #         try:
    #             json_match = re.search(r'\{.*\}', environment_response, re.DOTALL)
    #             if json_match:
    #                 json_data_e = json_match.group(0)
    #                 # Optionally parse the JSON here if you want consistency with mission_responses
    #                 parsed_json_e = json.loads(json_data_e)
    #                 environment_response_list.append(parsed_json_e)
    #             else:
    #                 print("No JSON found in environment response")
    #         except Exception as e:
    #             print(f"Error processing environment response: {e}")
                
    #     for mission in mission_responses:
    #         try:
    #             mission_json = re.search(r'\{.*\}', mission, re.DOTALL)
    #             if mission_json:
    #                 json_data_m = json.loads(mission_json.group(0))
    #                 mission_response_list.append(json_data_m)
    #             else:
    #                 print("No JSON found in mission response")
    #         except Exception as e:
    #             print(f"Error processing mission response: {e}")
    
    #     return environment_response_list, mission_response_list
    @staticmethod
    def RAGS_metrics(filename):
        data = pd.read_csv(filename)
        text = f"""
        prompt:- based on user Question  and RAGS Context and LLM response Give me following Score in the scale of 0-1
        user question:-{data["question"]}
        =======================================
        context:-{data["context"]}
        =======================================
        LLM Response :- {data["Senarios"]}

        Faithfulness - Measures the factual consistency of the answer to the context based on the question.

        Context_precision - Measures how relevant the retrieved context is to the question, conveying the quality of the retrieval pipeline.

        Answer_relevancy - Measures how relevant the answer is to the question.

        Context_recall - Measures the retriever’s ability to retrieve all necessary information required to answer the question.

        """
        messages = [ 
            {"role": "user", "content": "I am an Ai engineer need your Assistance"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 250
        agent_response = Agents.model_x(messages, temp, sample,tokens)
        pattern = r"(\d+)\.\s(\w+):\s(\d\.\d)\s-\s(.+?)\n"

        matches = re.findall(pattern, agent_response[-1]["content"])
        data = {
        "Category": [match[1] for match in matches],
        "Score": [match[2] for match in matches],
        "Explanation": [match[3] for match in matches]
    }

        # Create DataFrame
        df = pd.DataFrame(data)
        # print(df)
        # return agent_response[-1]["content"]
        return df
        # return "yep"


    @staticmethod
    def main(user_input,mission_type):
        scenario_response,cotext = Agents.scenario_generator_Agent(user_input)
        torch.cuda.empty_cache()
        missions, rest = Agents.helper_for_mission_and_environment(scenario_response)
        
        mission_responses = []
        environment_responses = []

        for mission in missions:
            mission_response = Agents.Mission_Agent(mission,mission_type)
            mission_responses.append(mission_response)
            torch.cuda.empty_cache()
        for rest_part in rest:
            environment_response = Agents.Environment_specification_Agent(rest_part)
            environment_responses.append(environment_response)
            torch.cuda.empty_cache()
        environment_json_list, mission_json_list,index = Agents.json_extraction(environment_responses, mission_responses)
        try:
            df1 = pd.DataFrame([user_input,cotext,scenario_response,str(index)])
            df1 = df1.T
            df1.columns = ["question","context","Senarios","index"]
            df2 = pd.DataFrame([mission_json_list,environment_json_list])
            df2 = df2.T
            df2.columns =["misson","environment"]
            df1.to_csv(f"user_questions/{user_input}_1.csv",index=False)
            df2.to_csv(f"user_questions/{user_input}_2.csv",index=False)
        except Exception as e:
            print(e)

        # try:
        #     rags_p = {
        # "scenario_response": scenario_response,
        # "context": context,
        # "mission_details": mission_json_list,
        # "environment_details": environment_json_list}

        #     df = pd.DataFrame(rags_p)
        #     df.to_csv(f"{user_input}.csv",index=False)
        return scenario_response ,cotext, mission_json_list, environment_json_list


    @staticmethod
    def Analytics_v1(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        text = f"""
        Drone Data logs :- {content}
        ===============================
        "Analyze the following drone log data and provide a detailed summary and key points. 
        The summary should include important events, notable patterns, and any anomalies. 
        Additionally, identify key metrics Present the information in a clear and concise manner, highlighting the most critical insights.
        """
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 1000
        agent_response = Agents.model_x(messages, temp, sample,tokens)
        return agent_response[-1]["content"]
    
    @staticmethod
    def Analytics_one(file_path):
        mission_data = pd.read_csv(file_path)
        # print(mission_data["Senarios"][0])
        # print("=======================================")
        text = f"""your An Analytics Agent designed to make what are appropriate Analytics required based on context given below
        Mission details :{mission_data["Senarios"][0]}
        make sure Analytics is inline with context and 100% required to have a look for root cause analysis
        instructions:
        1) keep in mind that you just have Access to Dron ulg data 
        2) we do not have access to drone camera data so 
        3) make sure the each senarios has different and Analytics that are relavent to senarios
        plan your Analytics Accordingly
        """
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 1000
        agent_response = Agents.model_x(messages, temp, sample,tokens)
        print(agent_response[-1]["content"])
        return agent_response[-1]["content"]

    @staticmethod
    def create_and_save_plots(data,filtered_data, output_folder):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for column in filtered_data:
            plt.figure()
            plt.plot(data.index, data[column], label=column)
            plt.xlabel('Index')
            plt.ylabel(column)
            plt.title(f'Plot of {column}')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_folder, f'{column}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved plot for {column} at {plot_path}')

    @staticmethod
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                img_path = os.path.join(folder, filename)
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except (IOError, OSError) as e:
                    print(f"Unable to open image {filename}: {e}")
        return images
        
    @staticmethod
    def Analytics_two(file_path):
        text = Agents.Analytics_one(file_path)
        torch.cuda.empty_cache()
        Analytics_list = []
        pattern = 'Scenario (\d+):\n([\s\S]*?)\n\n'
        matches = re.findall(pattern, text)
        
        for match in matches:
            Analytics_list.append(match)
        df = pd.read_csv("/home/uav/Documents/AI_Hunter/LLMS/ulg_data/test1.csv")
        non_empty_columns = [col for col in df.columns if df[col].notnull().any()]
        responses = []
        columns_l = []
        for i in range(0,len(non_empty_columns),250):
            #{Analytics_list[0][1]}
            prompt_t= f"""
            here are analytics i need for my root cause Analytics and
            "Analytics Topics"  = {Analytics_list[0][1]}
            ===============================
            "ulg_data columns":-{non_empty_columns[i:i+250]}
            and here are ulg data columns of my drone now tell me all what are columns required for above metrics
            rules:
            1)I have Given you "Analytics Topics" i need you to book contextually relevent "ulg_data columns" so that i can make better analytics of this data
        
            """
            # print(prompt_t)
            messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": prompt_t} ]
        
            temp = 0.0
            sample = False
            tokens = 1000
            agent_response = Agents.model_x(messages, temp, sample,tokens)
            responses.append(agent_response[-1]["content"])
            # return agent_response[-1]["content"]
        for i in range(len(responses)):
            pattern = r'-\s*(\S+)'
            extracted_values = re.findall(pattern, responses[i])
            columns_l.append(extracted_values)
        columns_l = [[item.strip("'") for item in sublist] for sublist in columns_l]#cleaning strings
        flattened_data = [item for sublist in columns_l for item in sublist]#flatten the list
        filtered_data = [item for item in flattened_data if item in non_empty_columns]#filter thedata
        Agents.create_and_save_plots(df,filtered_data,"plots_data")
        torch.cuda.empty_cache()
        #return responses

    @staticmethod
    def Analytics_three(file_path):#image_path):
        df = pd.read_csv("/home/uav/Documents/AI_Hunter/LLMS/ulg_data/test1.csv")
        # non_empty_columns = [col for col in df.columns if df[col].notnull().any()]
        # Agents.create_and_save_plots(df,filtered_data,"plots_data")
        # Agents.Analytics_two(file_path)
        image_path = input("Enter the path to the plots folder")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)#,llm_int4_threshold=6.0,llm_int8_skip_modules=None,trust_remote_code=True)
        
        model_id = "microsoft/Phi-3-vision-128k-instruct" 
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", quantization_config=quantization_config,trust_remote_code=True)

        # cls.model =  AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        
        images = Agents.load_images_from_folder(image_path)
        # s = ""
        # for i in range(len(images)):
        #     s+= f"<|image_{i+1}|>"


        responses_ll = []
        for i in range(0,len(images),5):
            image_tags = "".join([f"<|image_{j+1}|>" for j in range(len(images[i:i+5]))])

            promt_a = """based on these images give me detailed Analytics Report so that i get better understanding on my drone mission
            there are plots from my drone Ulg_data i have plotted them so now explain me what are these metrics  and how will these impact my drone behaviour
            rules:
            1)understand the plots and give me your opinions as a drone Expert
            2)keep the response as crisp summary and highlight soome key observations if need be
            3)if there is any correlations between plots highlite that
            """

            messages = [ 
                {"role": "user", "content":f"{image_tags}\nWhat is shown in this image?"}, 
                {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
                {"role": "user", "content": f"{promt_a}"} 
            ] 

            # image = Image.open(requests.get(url, stream=True).raw) 

            batch_images = images[i:i+5]
            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for j, image in enumerate(batch_images):
                image.id = j + 1
            inputs = processor(prompt,batch_images, return_tensors="pt").to("cuda:0") 

            generation_args = { 
                "max_new_tokens": 500, 
                "temperature": 0.0, 
                "do_sample": False, 
            } 

            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            responses_ll.append(response)
        Analysis = ""
        for res in responses_ll:
            Analysis += res

        print(f"Initial analysis is complete.\n{Analysis} You may now ask further questions.")
        while True:
            user_query = input("Enter your question or type 'exit' to finish: ")
            if user_query.lower() == 'exit':
                break
            prompt_Ab = f"""give me detailed Deep dive Analytics report based user question and Analysis Data
            Analysis data:-{Analysis}
            ==========================
            user questions :-{user_query}"""
            messages = [
                {"role": "system", "content": "intelligent AI system cabale of Answering any question you have"},  # Passing previous responses as memory
                {"role": "user", "content": f"{prompt_Ab}"}
            ]

            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt, return_tensors="pt").to("cuda:0")

            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("AI Response:", response)
        print("you need any questions further Apart from this Analytics??")
        while True:
            user_query = input("Enter your question or type 'exit' to finish: ")
            if user_query.lower() == 'exit':
                break
            cosine_similarity, indices = search_a(user_query, 5)
            contextlist = Analytics_data.iloc[indices[0]]["columns"]
            # context = combine_context(contextlist)
            Agents.create_and_save_plots(df,contextlist.to_list(),user_query)
            
            image_tags = "".join([f"<|image_{j+1}|>" for j in range(len(contextlist.to_list()))])
            path_to_load = input("give the path to directory")
            images_n = Agents.load_images_from_folder(path_to_load)
            promt_a = """based on these images give me detailed Analytics Report so that i get better understanding on my drone mission
            there are plots from my drone Ulg_data i have plotted them so now explain me what are these metrics  and how will these impact my drone behaviour
            rules:
            1)understand the plots and give me your opinions as a drone Expert
            2)keep the response as crisp summary and highlight soome key observations if need be
            3)if there is any correlations between plots highlite that
            """

            messages = [ 
                {"role": "user", "content":f"{image_tags}\nWhat is shown in this image?"}, 
                {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
                {"role": "user", "content": f"{promt_a}"} 
            ] 

            # messages = [
            #     {"role": "system", "content": f"{Analysis}"},  # Passing previous responses as memory
            #     {"role": "user", "content": f"{user_query}"}
            # ]

            prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(prompt,images_n, return_tensors="pt").to("cuda:0")

            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print("AI Response:", response)

        
            # return responses_ll
        # print(response)
        



# def parse_arguments():
#     parser = argparse.ArgumentParser(description="Automate UAV testing.")
#     parser.add_argument('--user_input', type=str, required=True, help='User input for scenario generation')
#     parser.add_argument('--mission_type', type=str, required=True, choices=['px4', 'drone_response'], help='Type of mission to generate JSON for')
#     return parser.parse_args()

if __name__ == "__main__":
    # rags_data = Agents.Analytics("/home/uav/Documents/AI_Hunter/LLMS/LLM_simulation/combined_texts.txt")
    # rags_data=Agents.main("i want more senarios for wind testing","px4")
    # rags_data = Agents.RAGS_metrics("/home/uav/Documents/AI_Hunter/LLMS/LLM_simulation/i want to test in high temparature and  no wind involvement senarios_1.csv")
    rags_data = Agents.Analytics_three("/home/uav/Documents/AI_Hunter/LLMS/LLM_simulation/Test the flight stability of my sUAS during river search and rescue operation_1.csv")
    print("rags performance metrics",rags_data)
    # args = parse_arguments()

    # scenario_response, context, mission_json_list, environment_json_list = Agents.main(args.user_input, args.mission_type)
    # print("Scenario Response:", scenario_response)
    # print("Context:", context)
    # print("Mission JSON List:", mission_json_list)
    # print("Environment JSON List:", environment_json_list)
