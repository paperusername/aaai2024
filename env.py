#  更换deepseek模型决策，固定金额流动，所有金额投资
import numpy as np
import torch
from LLMAgent import AdaptiveAgent
from Arguments import parse_arguments
import yaml
import json
from LLMPlanner import PlannerAgent
from typing import List, Dict, Callable
import random
import copy
from datetime import datetime
import pandas as pd


filepath = "data/"
featurepath = 'settings/agentinf.yaml'     #  feature settings
feature2path = 'settings/planner.yaml'   

Agent_financial_situation = ["normal" for _ in range(4)]     
Plannermat = {
    "Agent1": 
    {
        "loan_limit_for_Agent1": 20,
        "interest_rate_for_the_loan_to_Agent1": 0.1,
        "risk_level_Agent1": 1,
    },
    "Agent2": 
    {
        "loan_limit_for_Agent2": 20,
        "interest_rate_for_the_loan_to_Agent2": 0.1,
        "risk_level_Agent2": 3,
    },
    "Agent3": 
    {
        "loan_limit_for_Agent3": 20,
        "interest_rate_for_the_loan_to_Agent3": 0.1,
        "risk_level_Agent3": 3,
    },
}      
Plannermat = {
    "Agent1": 
    {
        "risk_level_Agent1": 1,
        "lending_amount_limit" : 20,
        "borrowing_interest_rate": 0.1,
        "lending_interest_rate": 0.05,
        "borrowing_counterpart": "nobody",
        "lending_counterpart" : "Agent2",
        "request_to_borrow_from_institution": True
    },
    "Agent2": 
    {
        "risk_level_Agent2": 1,
        "lending_amount_limit" : 20,
        "borrowing_interest_rate": 0.1,
        "lending_interest_rate": 0.05,
        "borrowing_counterpart": "nobody",
        "lending_counterpart" : "Agent3",
        "request_to_borrow_from_institution": True
    },
    "Agent3": 
    {
        "risk_level_Agent3": 1,
        "lending_amount_limit" : 20,
        "borrowing_interest_rate": 0.1,
        "lending_interest_rate": 0.05,
        "borrowing_counterpart": "nobody",
        "lending_counterpart" : "Agent2",
        "request_to_borrow_from_institution": True
    },
    "Agent4": 
    {
        "risk_level_Agent4": 1,
        "lending_amount_limit" : 20,
        "borrowing_interest_rate": 0.1,
        "lending_interest_rate": 0.05,
        "borrowing_counterpart": "nobody",
        "lending_counterpart" : "Agent3",
        "request_to_borrow_from_institution": True
    },
}   
Agent_match_matrix = {
    "Agent1": 
    {
      "lender": "nobody",
      "borrower": "nobody",
      "lend_rate": 0.15,
    },
    "Agent2": 
    {
      "lender": "nobody",
      "borrower": "nobody",
      "lend_rate": 0.15,
    },
    "Agent3": 
    {
      "lender": "nobody",
      "borrower": "nobody",
      "lend_rate": 0.15,
    },
        "Agent4": 
    {
      "lender": "nobody",
      "borrower": "nobody",
      "lend_rate": 0.15,
    },
  }     

def agentnames(n):
        agent_names = []
    for i in range(1, n+1):
        agent_name = f"Agent{i}"
        agent_names.append(agent_name)
        # print(agent_names)
    return agent_names

def genfeature(file, fea1):
    features = []
    feature = Datadeal.yaml_to_json(file)
    feature = json.loads(feature)       #  convert to json dict
    for i in range(len(fea1)):
        feature["preference"]['risk'] = fea1[i]
        feature1 = copy.deepcopy(feature)           #          
	features.append(feature1)         
    return features

def genfeature2(file, fea1, fea2):
    features = []
    feature = Datadeal.yaml_to_json(file)
    feature = json.loads(feature)       #  convert to json dict
    for i in range(len(fea1)):
        for j in range(len(fea2)):
            feature["preference"]["risk"] = fea1[i]
            feature["wealth"]["cash"] = fea2[j]
            feature1 = copy.deepcopy(feature)           #
	    features.append(feature1)     #  使用新的对象
    return features

def genplanner(file):
    feature = Datadeal.yaml_to_json(file)
    feature = json.loads(feature)       #  convert to json dict
    return feature

def replicate_elements(lst, n):
    # 使用列表推导式来复制每个元素 n 次
    new_list = [item for item in lst for _ in range(n)]
    return new_list

def select_speaker_order(n):
    # idx = random.randint(0, n-1)
    # print('id: ', idx)
    # # print('select name: ', names[idx])
    # print('')
    idx = list(range(1, n+1))
    random.shuffle(idx)
    return idx


class Datadeal:


    def yaml_to_json(yaml_str):
        with open(yaml_str, 'r') as file:
            yaml_data = yaml.safe_load(file)
        json_data = json.dumps(yaml_data, indent=2)
        return json_data


    def json_to_yaml(json_str):
        with open(json_str, 'r') as file:
            json_data = json.loads(file)
        yaml_data = yaml.dump(json_data, default_flow_style=False)
        return yaml_data
    

    def save_json_data(json_data):
        with open(filepath+"output.json", "w") as json_file:
            json_file.write(json_data)


    def save_yaml_data(yaml_data):
        with open(filepath+"output.yaml", "w") as yaml_file:
            yaml_file.write(yaml_data)

#  the observation is list of numpy
class MultiAgentEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.currentstep = 0
        self.savenum = 0    #  记录保存state顺序
        self.agent_states = {}   #  放入所有agents状态
        self.agentrisk = {}     #  记录所有智能体的风险评级


        self.chathistory =  ["Here is the conversation so far."]      #  记录所有智能体对话历史（无regulator）
        self.summaryhistory = ["Here is the summary so far. "]      #  记录所有智能体总结历史（无regulator）
        self.envhistory = ["Here is the environment dynamic so far. "]  #  记录环境变化历史（无regulator）
        self.policyhistory = ["Here are the policies of the agents so far."]        #  记录策略变化
        self.plannerpolicyhistory = ["Here are the policies of the social planner so far."]     #  记录regulator策略变化
        self.plannersummaryhistory = ["Here is the summary so far. "]  #  记录regulator总结历史
        self.disclosureinformation = ["Here is the information of disclosure. "]    #  记录regulator建议披露信息
        self.policytrends = ["Here is the policy trend announced by regulator. "]  #  记录regulator政策趋势

        #  智能体特征
        prefer_risk = ["risk-averse", "risk-loving"]        #  风险偏好
        cash = [10, 20]        #  现金禀赋差异
        features = genfeature2(featurepath, prefer_risk, cash)     #  生成所有类别的特征
        # features = genfeature(featurepath, prefer_risk)     #  生成所有类别的特征
        # print('features: ', features)
        self.features = features
        self.socialplannermsg = []      #  定义列表存放 regulator 的披露消息！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！重要！！！！！！！！！！！！！！！！！！！！！！！！！！！======
        self.socialplannersave = []     #  定义列表存放 regulator 的全部披露消息！！！！！！！！！！！！！！！！！！

        self.num_agents_class = int(num_agents/(len(prefer_risk)*len(cash)))       #  每一类的智能体数量
        print('number of each type of agent: ', self.num_agents_class)
        self.all_agents_feature = replicate_elements(features, self.num_agents_class)       #  所有智能体特征
        # print('features: ',features)
        self.ids = list(range(1, self.num_agents+1))     #  生成所有agent id, 从1到n.
        self.names = agentnames(self.num_agents)        #  生成所有agents姓名
        print('\n\n All agents` names: ', self.names)
        # print('self.all_agents_feature: ', self.all_agents_feature)
        all_agents_feature = []     #   ====================================================================================
        for i in range(self.num_agents):
            # print('i = ', i)
            # print('self.name[i]: ', self.names[i])
            # print("all_agents_feature[i]: ", self.all_agents_feature[i])
            agent_i_feature = copy.deepcopy(self.all_agents_feature[i])
            agent_i_feature['name'] = self.names[i]      #  更新智能体参数=======================================================================================
            agent_i_feature['namelist'] = self.names      #  更新所有agent的名单list
            all_agents_feature.append(copy.deepcopy(agent_i_feature))
        self.all_agents_feature = copy.deepcopy(all_agents_feature)

        print('All agents` features: ', self.all_agents_feature)

        self.agents = [AdaptiveAgent(state_dim, action_dim, features) for features in zip(self.all_agents_feature)]

        #  规划者特征
        featureplanner = genplanner(feature2path)
        print('featureplanner: ', featureplanner)
        self.planner = PlannerAgent(state_dim, action_dim, featureplanner)


    def save_load_chat(self, filename, content, action):       #  保存对话历史
        # 获取当前时间
        current_time = datetime.now()
        # 将时间格式化为字符串，例如：2024-01-31_12-30-45
        formattime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = filename.split('.json')[0]+'_'+formattime+'.json'
        if action == 'save':
            # save json file
            output_file_path = filename
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(content, output_file, ensure_ascii=False, indent=2)
            data = []
        if action == 'load':
            with open(filename, 'r') as file:
                data = json.load(file)      # 使用json.load()加载JSON数据
        return data
 
 
    def chatstep(self) -> tuple[str, str]:      #  一方广播，系统根据规则匹配==============================================================================================================
        # 1. choose the next speaker
        # print('num of agents: ', self.num_agents)
        speaker_idx = select_speaker_order(self.num_agents)
        print("speaker id: ", speaker_idx)
        envdescrip = [f"This is the recent situation in the environment."]       #  对环境的描述========================================================================
        for _ in range(2):
            for k in range(self.num_agents):
                print('id: ', speaker_idx[k])
                speaker = self.agents[speaker_idx[k]-1]     #  id转变成序号
                # print("speaker: ", speaker)
        
                # 2. next speaker sends message (policy and description)

                msgpolicy, msgchat = speaker.send(self.chathistory, self.socialplannermsg, word_limit)
                print("step = "+str(self.currentstep) + '\n' + speaker.name+': \n', msgchat)

                self.policyhistory.append(msgpolicy)        #  更新智能体策略历史

                #  更新群体特征矩阵
                agentformat = json.loads(msgpolicy)      #  转化成字典格式
                # print('agentformat keys: ', agentformat.keys())
                mat_borrow_agent = agentformat['borrowing_counterpart']       #  提取策略值
                mat_lend_agent = agentformat["lending_counterpart"]
                mat_lend_rate = agentformat["lending_interest_rate"]
                Agent_match_matrix[speaker.name]["borrower"] = mat_borrow_agent     #  更新群体数据，备传regulator
                Agent_match_matrix[speaker.name]["lender"] = mat_lend_agent
                Agent_match_matrix[speaker.name]["lend_rate"] = mat_lend_rate
                speaker.lendrate = mat_lend_rate        #  给期望借款利率赋值

                if agentformat["request_to_borrow_from_institution"] == True:       #  向银行借钱次数+1
                    speaker.num_borrow_planner = speaker.num_borrow_planner + 1

                # #  判断是否破产
                # if speaker.cash < 0:
                #     speaker.financial_situation = "bankrupt"
                # Agent_financial_situation[speaker_idx[k]-1] = speaker.financial_situation        #  更新财务状态

                #  更新智能体策略（处理结构化数据，转化为输入）+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                envtransition = speaker.update(self.currentstep, Agent_match_matrix, Agent_financial_situation, Plannermat)
                envdescrip.append(envtransition)        #  返回环境描述！！！！！！重要，重要！！！！！！！！！！！！！！

                self.chathistory.append(f"step = {self.currentstep}, {speaker.name}: {msgchat}")        #  更新记录对话历史
                # self.chathistory = self.chathistory + [f"{speaker.name}: {msgchat} \n"]     

                thought = speaker.summary(self.chathistory, word_limit)
                print(speaker.name+'`s summary: \n', thought)

                self.summaryhistory.append(f"step = {self.currentstep}, {speaker.name}: {thought}")        #  更新记录总结历史

                self.envhistory.append(envdescrip)      #  更新记录环境动态

                # print('speaker state: ', speaker.state)

                self.savenum = self.savenum + 1
                self.agent_states[self.savenum] = speaker.state     #  记录agent状态==========================================================

        #  planner的部分
        plannersummary = self.planner.planner_summary(self.policyhistory, envdescrip, word_limit)       #  生成social planner总结建议
        plannerpolicy = self.planner.regulatordecsion(self.policyhistory, plannersummary, speaker.invest_rate, envdescrip)        #  加载智能体新策略，生成策略
        # print('social planner`s policy: ', plannerpolicy)
        # plannerpolicy = self.planner.planner_description(self.policyhistory, plannersummary, speaker.invest_rate, envdescrip)        #  加载智能体新策略，生成策略
        # print('social planner`s policy: ', plannerpolicy)
        plannerformat = json.loads(plannerpolicy)       #  生成结构化的策略
        print("plannerformat: ", plannerformat)     

        disclosure, risk_rating, trend = self.planner.planner_macro_adjust(self.policyhistory, envdescrip, word_limit)       #  生成披露信息和政策走向 ！！重要！！！！！！！！！！！！！！
        self.socialplannermsg = [disclosure, risk_rating, trend]        #  更新social planner信息
        self.socialplannersave.append(self.socialplannermsg)        #  更新social planner信息库

        self.plannersummaryhistory.append(f"""Step = {self.currentstep}: {plannersummary}""")   #  更新social planner总结历史
        self.plannerpolicyhistory.append(f"""Step = {self.currentstep}: {plannerpolicy}""")     #  更新social planner策略历史

        #  Plannermat待更新（planner传给agent）+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Plannermat["Agent1"]["loan_limit_for_Agent1"] = plannerformat["loan_limit_for_Agent1"]
        # Plannermat["Agent1"]["interest_rate_for_the_loan_to_Agent1"] = plannerformat["interest_rate_for_the_loan_to_Agent1"]
        Plannermat["Agent1"]["risk_level_Agent1"] = plannerformat["risk_level_Agent1"]
        Plannermat["Agent2"]["loan_limit_for_Agent2"] = plannerformat["loan_limit_for_Agent2"]
        # Plannermat["Agent2"]["interest_rate_for_the_loan_to_Agent2"] = plannerformat["interest_rate_for_the_loan_to_Agent2"]
        Plannermat["Agent2"]["risk_level_Agent2"] = plannerformat["risk_level_Agent2"]
        Plannermat["Agent3"]["loan_limit_for_Agent3"] = plannerformat["loan_limit_for_Agent3"]
        # Plannermat["Agent3"]["interest_rate_for_the_loan_to_Agent3"] = plannerformat["interest_rate_for_thecler_loan_to_Agent3"]
        Plannermat["Agent3"]["risk_level_Agent3"] = plannerformat["risk_level_Agent3"]

             self.planner.update(Plannermat)      #  重要！！！整合所有agent借款



        chatpath = "data/History/chatlog.json"
        _ = self.save_load_chat(chatpath, self.chathistory, 'save')        #  保存对话历史信息    
        # self.chathistory = self.save_load_chat(chatpath, self.chathistory, 'load')    #  加载
        policypath = "data/History/policy.json"
        _ = self.save_load_chat(policypath, self.policyhistory, 'save')        #  保存environment dynamic  
        summarypath = "data/History/summarylog.json"
        _ = self.save_load_chat(summarypath, self.summaryhistory, 'save')        #  保存summary       
        envlogpath = "data/History/envlog.json"
        _ = self.save_load_chat(envlogpath, self.envhistory, 'save')        #  保存environment dynamic     
        plannerpolicypath = "data/History/plannerpolicylog.json"
        _ = self.save_load_chat(plannerpolicypath, self.plannerpolicyhistory, 'save')        #  保存planner policy
        plannersummarypath = "data/History/plannersummarylog.json"
        _ = self.save_load_chat(plannersummarypath, self.plannersummaryhistory, 'save')        #  保存planner summary           
        plannerdisplosurepath = "data/History/plannerdisplosurelog.json"
        _ = self.save_load_chat(plannerdisplosurepath, self.socialplannersave, 'save')        #  保存planner summary
        statespath = "data/History/stateslog.csv"
        # with open(statespath, 'w') as json_file:
        #     json.dump(self.agent_states, json_file)        
        pd.DataFrame(self.agent_states).to_csv(statespath, index=False)        #  保存agent 的状态           


        # 5. increment time
        self.currentstep += 1
 
        return speaker.name, msgchat, self.currentstep
        
    def reset(self):
        # 重置智能体
        for agent in self.agents:
            agent.reset()
        # 重置环境并返回初始状态
        self.planner.reset()        #  初始化social planner
        self.state = np.random.rand(self.num_agents, self.state_dim)        #  state获取
        return self.state
    
if __name__ == "__main__":
    
    # 创建环境
    num_agents = 4
    state_dim = 5
    action_dim = 2
    environment = MultiAgentEnvironment(num_agents, state_dim, action_dim)

    word_limit = 160     #  设置输出语句长度限制
        
    # descrip_format  = environment.agents[0].description()
    # print("descrip_format: ",  descrip_format)

    environment.reset()
    for i in range(16):
        environment.chatstep()