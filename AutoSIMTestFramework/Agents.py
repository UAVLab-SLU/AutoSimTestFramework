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
import matplotlib.pyplot as plt
import json
from groq import Groq

import argparse
data = pd.read_csv("knowledga_base.csv")
root = '/Users/aadityashah/Downloads/LLMS/'

Analytics_data = pd.read_csv(f"{root}/ulg_data/wind1_context_columns.csv")

with open('missions.json', 'r') as file:
    mission_json_data = json.load(file)


with open(f'{root}/columns.txt', 'r') as file:
    analytics_columns = file.read()

#load knowledge base
knowledge = data.drop(["filename","synopsis"],axis=1)
knowledge_tensor = torch.tensor(knowledge.values, dtype=torch.float32)
normalized_embeddings  = F.normalize(knowledge_tensor, p=2.0, dim=1)
dimension = knowledge_tensor.shape[1]
index = faiss.IndexFlatIP(dimension)

index.add(normalized_embeddings.numpy())

##load the Analytics
knowledge_a = Analytics_data.drop(["parameter","description","content"],axis=1)
knowledge_tensor_a = torch.tensor(knowledge_a.values, dtype=torch.float32)
normalized_embeddings_a  = F.normalize(knowledge_tensor_a, p=2.0, dim=1)
dimension_a = knowledge_tensor_a.shape[1]
index_a = faiss.IndexFlatIP(dimension_a)

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
    return "---\n".join(contentlist)

class ModelManager:
    def __init__(self):
        self.groq_client = Groq(api_key="YOUR API KEY HERE")

    def load_model(self, model_id):
        # Only Groq models are supported now
        return None, None, None

class Agents:
    model_manager = ModelManager()
    
    @staticmethod
    def model_x(messages, temp, sample, tokens, model_id):
        # Format messages for Groq
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
                
        completion = Agents.model_manager.groq_client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            temperature=temp,
            max_tokens=tokens
        )
        return [{"content": completion.choices[0].message.content}]

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
        response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
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
        agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
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
        agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
        return agent_response[-1]["content"]
        
    @staticmethod
    def json_extraction(environment_responses, mission_responses):
        environment_response_list, mission_response_list,index = [], [],[]
    
        # Assuming environment_responses and mission_responses are of the same length
        for i in range(len(environment_responses)):
            environment_response = environment_responses[i]
            mission_response = mission_responses[i]
            try:
                # Extract and parse environment JSON
                json_match_env = re.search(r'\{.*\}', environment_response, re.DOTALL)
                if json_match_env:
                    json_data_e = json.loads(json_match_env.group(0))
                else:
                    raise ValueError("No JSON found in environment response")
    
                # Extract and parse mission JSON
                json_match_mission = re.search(r'\{.*\}', mission_response, re.DOTALL)
                if json_match_mission:
                    json_data_m = json.loads(json_match_mission.group(0))
                else:
                    raise ValueError("No JSON found in mission response")
    
                # If both extractions are successful, append to lists
                environment_response_list.append(json_data_e)
                mission_response_list.append(json_data_m)
                index.append(i)
    
            except Exception as e:
                # If an error occurs in either, skip both
                print(f"Error processing pair at index {i}: {e}")
    
        return environment_response_list, mission_response_list,index


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

        Context_recall - Measures the retriever's ability to retrieve all necessary information required to answer the question.

        """
        messages = [ 
            {"role": "user", "content": "I am an Ai engineer need your Assistance"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 250
        agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
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
        missions, rest = Agents.helper_for_mission_and_environment(scenario_response)
        
        mission_responses = []
        environment_responses = []

        for mission in missions:
            mission_response = Agents.Mission_Agent(mission,mission_type)
            mission_responses.append(mission_response)
        for rest_part in rest:
            environment_response = Agents.Environment_specification_Agent(rest_part)
            environment_responses.append(environment_response)
        environment_json_list, mission_json_list,index = Agents.json_extraction(environment_responses, mission_responses)
        try:
            df1 = pd.DataFrame([user_input,cotext,scenario_response,str(index)])
            df1 = df1.T
            df1.columns = ["question","context","Senarios","index"]
            df2 = pd.DataFrame([mission_responses,environment_responses])
            df2 = df2.T
            df2.columns =["misson","environment"]
            df1.to_csv(f"user_questions/{user_input}_1.csv",index=False)
            df2.to_csv(f"user_questions/{user_input}_2.csv",index=False)
        except Exception as e:
            print(e)
        m_r = combine_context(mission_responses)
        e_r = combine_context(environment_responses)
        # try:
        #     rags_p = {
        # "scenario_response": scenario_response,
        # "context": context,
        # "mission_details": mission_json_list,
        # "environment_details": environment_json_list}

        #     df = pd.DataFrame(rags_p)
        #     df.to_csv(f"{user_input}.csv",index=False)
        return scenario_response ,cotext,m_r,e_r# mission_responses, environment_responses


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
        agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
        return agent_response[-1]["content"]
    @staticmethod
    def extract_scenarios(text):
        pattern = r'(Scenario \d+:\n\n[\s\S]*?)(?=Scenario \d+:|$)'
        matches = re.findall(pattern, text)
        scenarios = [match.strip() for match in matches]
        return scenarios
    @staticmethod
    def Analytics_one(file_path):

        mission_data = pd.read_csv(f"user_questions/{file_path}_1.csv")
        mission_scenarios = Agents.extract_scenarios(mission_data["Senarios"][0])
        # print(mission_data["Senarios"][0])
        # print("=======================================")
        text = f"""your An Analytics Agent designed to make what are appropriate Analytics required based on context given below
        Mission details :{mission_scenarios[1]}
        make sure Analytics is inline with context and 100% required to have a look for root cause analysis
       instructions:-
       1) mission details is for your understanding so use that as just  context and give Analytics that can make me understand complete drone mission
       so that next time when i create senarios i will keep that in mind
       2) make sure your cover folling metrics 
        1) Environmental Analysis 2)Obstacle Analysis 3)Sensor Performance 4)Communication Analysis 5)Mission Planning 6)Root Cause Analysis 7)Performance Metrics

        """
        messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": text} 
        ]
        
        temp = 0.0
        sample = False
        tokens = 1000
        agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
        print(agent_response[-1]["content"])
        return agent_response[-1]["content"]

    @staticmethod
    def create_and_save_plots(data,filtered_data, output_folder):

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        time = pd.to_datetime(data['timestamp'], unit='us')

        for column in filtered_data:
            plt.figure()
            # column_data = data[column].dropna().reset_index(drop=True)
            # plt.plot(column_data.index, column_data, label=column)
            # plt.plot(data.index, data[column], label=column)
            plt.plot(time, data[column], label=column)

            plt.xlabel('Time')
            plt.ylabel(column)
            plt.title(f'Plot of {column}')
            plt.legend()
            plt.grid(True)
            # plt.ylim(bottom=0)
            plt.xlim(left=min(time),right=max(time))
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
        


    # Analytics_list = []
        # # pattern = 'Scenario (\d+):\n([\s\S]*?)\n\n'
        # pattern = r'(Scenario \d+: [\s\S]*?)(?=Scenario \d+:|$)'
        # matches = re.findall(pattern, text)
        
        # for match in matches:
        #     Analytics_list.append(match)
    @staticmethod
    def Analytics_two(file_path):
        text = Agents.Analytics_one(file_path)
        
        df = pd.read_csv(f"{root}/ulg_data/wind1_filtered.csv")
        non_empty_columns = [col for col in df.columns if df[col].notnull().any()]
        responses = []
        columns_l = []
        for i in range(0,len(non_empty_columns),250):
            #{Analytics_list[0][1]}
            prompt_t= f"""
            here are analytics i need for my root cause Analytics and
            "Analytics Topics"  = {text}
            ===============================
            "ulg_data columns":-{non_empty_columns[i:i+250]}
            and here are ulg data columns of my drone now tell me all what are columns required for above metrics
            
            rules:
            1)I have Given you "Analytics Topics" i need you to look contextually relevent "ulg_data columns" so that i can make better analytics of this data
            2)Select only the columns that will enhance the quality and accuracy of the analytics.
            3)Ensure the selection is comprehensive but not redundant.
            4) just give column names no need for description and column names as same as given input
            """
            # print(prompt_t)
            messages = [ 
            {"role": "user", "content": "I am an sUAS Software designer and I need your assistance on Automating UAV testing"}, 
            {"role": "assistant", "content": "I am an AI system capable of answering any query you have"}, 
            {"role": "user", "content": prompt_t} ]
        
            temp = 0.0
            sample = False
            tokens = 1000
            agent_response = Agents.model_x(messages, temp, sample,tokens,"llama-3.3-70b-versatile")
            responses.append(agent_response[-1]["content"])
            torch.cuda.empty_cache()
            # return agent_response[-1]["content"]
        print(responses)
        for i in range(len(responses)):
            pattern = r'-\s*(\S+)'
            extracted_values = re.findall(pattern, responses[i])
            columns_l.append(extracted_values)
        print(columns_l)
        columns_l = [[item.strip("'") for item in sublist] for sublist in columns_l]#cleaning strings
        flattened_data = [item for sublist in columns_l for item in sublist]#flatten the list
        filtered_data = [item for item in flattened_data if item in non_empty_columns]#filter thedata
        Agents.create_and_save_plots(df,filtered_data,file_path)
        images = Agents.load_images_from_folder(file_path)
        torch.cuda.empty_cache()
        return text,images
        #return responses

    @staticmethod
    def Analytics_three(file_path):#image_path):
        # df = pd.read_csv(f"{root}/ulg_data/wind1_filtered.csv")
        
        df = pd.read_csv(f"{root}/ulg_data/06_40_19.csv")
        # non_empty_columns = [col for col in df.columns if df[col].notnull().any()]
        # Agents.create_and_save_plots(df,filtered_data,"plots_data")
        # text = Agents.Analytics_one(file_path)
        text,images = Agents.Analytics_two(file_path)
        # image_path = input("Enter the path to the plots folder")
        
        responses_ll = []
        for i in range(0,len(images),5):
            image_tags = "".join([f"<|image_{j+1}|>" for j in range(len(images[i:i+5]))])

            promt_a = """
            Analytics Report Request for Drone ULG Data
            I have a set of plots derived from my drone's ULG data. As an expert in drone analytics, I would like you to analyze these plots and provide a detailed report.
            The report should include the following:
            1)Understanding the Plots 2)Impact on Drone Behavior 3)Key Observations 4)Correlations Between Plots
            Rules:
            1)Understand the plots and provide your expert opinion.
            2)Keep the response crisp and summarized, with key observations highlighted.
            3)Highlight any correlations between plots.
            Please ensure the report is comprehensive yet concise, offering actionable insights and a clear understanding of the drone's performance based on the ULG data.
            """

            messages = [ 
                {"role": "user", "content":f"{image_tags}\ni would like a detailed analysis of the images I have provided, focusing on the metrics displayed and their impact on drone behavior"}, 
                {"role": "assistant", "content": "understand what is present in images and give very insightfull responses as per user questions"}, 
                {"role": "user", "content": f"{promt_a}"} 
            ]

            temp = 0.0
            sample = False
            tokens = 1000
            agent_response = Agents.model_x(messages, temp, sample, tokens, "llama-3.3-70b-versatile")
            responses_ll.append(agent_response[-1]["content"])
            torch.cuda.empty_cache()

        Analysis = ""
        for res in responses_ll:
            Analysis += res
        
        df_t = pd.DataFrame(responses_ll)
        df_t.to_csv(f"user_analytics/{file_path}.csv",index =False)
        print(Analysis)
        return text,images,Analysis

    @staticmethod
    def clarification(file_path,user_query):
        Data_a = pd.read_csv(f"user_analytics/{file_path}.csv")

        Analysis = ""
        for res in Data_a["0"]:
            Analysis += res

        prompt_Ab = f"""
        Provide explanation and clarification based on the user's question regarding the provided analytics data.
        ==========================
        Analysis data:-{Analysis}
        ==========================
        instructions:-
        user questions :-{user_query}"""
        messages = [
            {"role": "system", "content": "intelligent AI system cabale of Answering any question you have"},
            {"role": "user", "content": f"{prompt_Ab}"}
        ]

        temp = 0.0
        sample = False
        tokens = 1000
        agent_response = Agents.model_x(messages, temp, sample, tokens, "llama-3.3-70b-versatile")
        return agent_response[-1]["content"]

    @staticmethod
    def new_analytics(user_query):
        df = pd.read_csv(f"{root}/ulg_data/06_40_19.csv")
        cosine_similarity, indices = search_a(user_query, 5)
        contextlist = Analytics_data.iloc[indices[0]]["parameter"]
        Agents.create_and_save_plots(df,contextlist.to_list(),user_query)
        
        image_tags = "".join([f"<|image_{j+1}|>" for j in range(len(contextlist.to_list()))])
        path_to_load = user_query
        images_n = Agents.load_images_from_folder(path_to_load)
        promt_a = """
        Analytics Report Request for Drone ULG Data
        I have a set of plots derived from my drone's ULG data. As an expert in drone analytics, I would like you to analyze these plots and provide a detailed report.
        The report should include the following:
        1)Understanding the Plots 2)Impact on Drone Behavior 3)Key Observations 4)Correlations Between Plots
        Rules:
        1)Understand the plots and provide your expert opinion.
        2)Keep the response crisp and summarized, with key observations highlighted.
        3)Highlight any correlations between plots.
        Please ensure the report is comprehensive yet concise, offering actionable insights and a clear understanding of the drone's performance based on the ULG data.
        """

        messages = [ 
            {"role": "user", "content":f"{image_tags}\ni would like a detailed analysis of the images I have provided, focusing on the metrics displayed and their impact on drone behavior"}, 
            {"role": "assistant", "content": "understand what is present in images and give very insightfull responses as per user questions"}, 
            {"role": "user", "content": f"{promt_a}"} 
        ]

        temp = 0.0
        sample = False
        tokens = 1000
        agent_response = Agents.model_x(messages, temp, sample, tokens, "llama-3.3-70b-versatile")
        return images_n, agent_response[-1]["content"]


# if __name__ == "__main__":
 
#     rags_data,iamges = Agents.Analytics_three("i want scenarios that involve very high wind conditions. i wan to test my system more in the context of wind so make sure i cover all edge cases in that conext")
#     print("rags performance metrics",rags_data)
