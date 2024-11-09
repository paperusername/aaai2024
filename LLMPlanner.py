#  能顺利运行，结构化生成决策数据，引入socialplanner耦合决策
from openai import OpenAI

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import json

client = OpenAI(
    api_key=API_KEY_ds,
    base_url=URLds
) 


def getresponse(prompt, t, temperature, model):
    while True:
        try:
            # time.sleep(t)
            completion = client.chat.completions.create(
            model = model,  
            messages = [    
                {"role": "system", "content": prompt[0]},    
                {"role": "user", "content": prompt[1]} 
            ],
            temperature = temperature
            ) 

            response = completion.choices[0].message.content
            # print(response)
            return response 
            break
        except Exception as e:
            # 出错时打印异常信息（可选）
            print(f"Error: {e}")


class PlannerAgent(nn.Module):
    def __init__(self, input_size, output_size, features):
        super(PlannerAgent, self).__init__()
 
        #  定义特征
        self.name = 'Regulator'
        self.agent_states = 'Agent states'      #  初始化所有观测的智能体状态
        self.features = features
        # print('planner feature: ', features)
        # print('features: ', features)
        self.namelist = features['namelist']
        self.totalsteps = features['interaction']['total_steps']
        self.step = features['interaction']['step']

        self.thought = ''   #  定义总结语句库

        # self.system_message = system_message        #  定义系统消息
        # self.system_message_format = system_message_format

        self.prefix = f"{self.name}: "
        self.reset()

        self.lendlimit = 300        #  借款总限额
        self.limit = {"loan_limit_for_Agent1": 100, "loan_limit_for_Agent2": 100,"loan_limit_for_Agent3": 100}     #  当前借款限额
        self.plannerpolicy = ""     #  记录生成策略
        self.investrate = 0     #  记录投资利率
        self.invest_rate = [0.05, 0.1]      #  智能体投资两类资产的利率
        self.unitcash = 10      #  单位现金量
        self.stateall = []      #  所有智能体的信息=======================================================================================
        self.risklevels = []    #  所有智能体的风险评级======================================================================================
        self.investrate3 = 0.08

    def reset(self):
        self.message_history = ["Here is the conversation so far."]
        self.stateall = ["Here are the information of all the agents."]

    def update(self, Plannermat):      #   [social planner决策矩阵，所有人借款信息list]
        self.limit["loan_limit_for_Agent1"] = Plannermat["Agent1"]["loan_limit_for_Agent1"]
        self.limit["loan_limit_for_Agent2"] = Plannermat["Agent2"]["loan_limit_for_Agent2"]
        self.limit["loan_limit_for_Agent3"] = Plannermat["Agent3"]["loan_limit_for_Agent3"]

    def send(self, chathistory, word_limit) -> str:

        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        self.message_history = chathistory
        #  生成智能体策略 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++补充环境观察
        while True:
            try:
                descrip_format  = self.description()
                # print('Policy: ', descrip_format)
                #  生成借款对话
                # borrow_name = self.get_specific_item(descrip_format, item = "[borrowing_agent]")
                descrip_format_data = json.loads(descrip_format)
                # print('descrip_format_data: ', descrip_format_data)
                borrow_name = descrip_format_data['borrowing_agent']
                lender_name = descrip_format_data['lending_agent']
                # print(borrow_name)
                break
            except json.JSONDecodeError as e:
                NotImplementedError
                # print(f"JSON decoding error: {e}")
        # print('borrow_name: ', borrow_name)
        borrower_chat = self.generate_chat_description(descrip_format, borrow_name, lender_name, word_limit)
        # print('Chat to borrower: ', borrower_chat)

        return descrip_format, borrower_chat   #  policy message and chatmessage separately

    def receive(self, name: str, message: str) -> None:
 
        """
        Concatenates {message} spoken by {name} into message history
        """
 
        self.message_history.append(f"{name}: {message}")

    def receive_load_save(self, name: str, message: str) -> None:       #  从库中加载和保存
 
        """
        Concatenates {message} spoken by {name} into message history
        """
 
        self.message_history.append(f"{name}: {message}")

        # # save json file
        # chatsavepath = "data/History/ChatHistory/"       #   增加文件名变量==============================================================================================================
        # output_file_path = chatsavepath + "chatlog.json"
        # with open(output_file_path, "w", encoding="utf-8") as output_file:
        #     json.dump(self.message_history, output_file, ensure_ascii=False, indent=2)
            
    #  [policy]  根据智能体信息和自己的目标决定利率(social planner)
    def planner_description(self, agentstate, planner_summary, investrate, envdescription):
        self.agent_states = agentstate
        self.investrate = investrate
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to reduce the bankruptcy rate of all the agents ({self.namelist}) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is {self.lendlimit} units.
        You decide on the individual borrowing limits for {self.namelist}. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        The content is :

        {self.stateall}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [borrowing_interest_rate]: the borrowing interest rate to other agents. 
        [lending_interest_rate]: the lending interest rate to other agents. 
        [Investigation]: whether to invest. Its value is true or false.
        [investasset]: you can choose one from Asset1 and Asset2. Investing in Asset1 provides a fixed return rate of {self.invest_rate[0]} per step; investing in Asset2 yields an average return rate of {self.invest_rate[1]} per step, with a 50% probability of a return rate of -100% and a 0.5 probability of a return rate of {(1+2*self.invest_rate[1])*100}%. 
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_institution]: whether to borrow money from a third party, a unique choice must be made between true and false. The interest rate for borrowing from a third party is {self.investrate3}, with a borrowing limit set by the third party of {self.limit3}, determined by factors such as risk rating, information disclosure, and debt situation.
        [borrowing_counterpart]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_counterpart] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 

        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        You can start by conducting a risk assessment and then optimize the maximum borrowing amount (loan limit) for each agent based on the assessment results.
              
        [loan_limit_for_Agent1], [loan_limit_for_Agent2], [loan_limit_for_Agent3] means the upper limit of the amount you lend to each agent. The sum of them is less than {self.lendlimit}.
        [risk_level_Agent1], [risk_level_Agent2], [risk_level_Agent3] denotes the coefficient indicating the potential occurrence of risks by an agent, with values ranging from [1, 2, 3], where higher numerical values correspond to elevated risk levels. You can only choose one.
        
        Please replan according to the chat history and your thought. The output format is JSON, the JSON file includes "loan_limit_for_Agent1", "loan_limit_for_Agent2", "loan_limit_for_Agent3", "risk_level_Agent1", "risk_level_Agent2", "risk_level_Agent3". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content."""

        history = f"""

        The chat history:

        {self.message_history}

        Latest developments:

        {envdescription}

        My summary:

        {planner_summary}

        Template:

        "loan_limit_for_Agent1": 100
        "loan_limit_for_Agent2": 100
        "loan_limit_for_Agent3": 100
        "loan_limit_for_Agent4": 100
        "risk_level_Agent1": 1
        "risk_level_Agent2": 2
        "risk_level_Agent3": 3
        "risk_level_Agent4": 5
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to lend, deciding on the recipients and amounts of your loans, setting the loan interest rates, and actively initiating interactions. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)
        self.plannerpolicy = descrip        #       保存到自有策略

        return descrip
    
    #  [policy]  根据智能体信息和自己的目标决定风险评级和分配(regulator), 信息披露
    def regulatordecsion(self, agentstate, planner_summary, investrate, envdescription):
        self.agent_states = agentstate
        self.investrate = investrate
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to reduce the bankruptcy rate of all the agents ({self.namelist}) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is {self.lendlimit} units.
        You decide on the individual borrowing limits for {self.namelist}. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        The content is :

        {self.stateall}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [borrowing_interest_rate]: the borrowing interest rate to other agents. 
        [lending_interest_rate]: the lending interest rate to other agents. 
        [Investigation]: whether to invest. Its value is true or false.
        [investasset]: you can choose one from Asset1 and Asset2. Investing in Asset1 provides a fixed return rate of {self.invest_rate[0]} per step; investing in Asset2 yields an average return rate of {self.invest_rate[1]} per step, with a 50% probability of a return rate of -100% and a 0.5 probability of a return rate of {(1+2*self.invest_rate[1])*100}%. 
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_institution]: whether to borrow money from a third party, a unique choice must be made between true and false. The interest rate for borrowing from a third party is {self.interestrate3}, with a borrowing limit set by the third party of {self.limit3}, determined by factors such as risk rating, information disclosure, and debt situation.
        [borrowing_counterpart]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_counterpart] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 

        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        You can start by conducting a risk assessment and then optimize the maximum borrowing amount (loan limit) for each agent based on the assessment results.
              
        [loan_limit_for_Agent1], [loan_limit_for_Agent2], [loan_limit_for_Agent3] means the upper limit of the amount you lend to each agent. The sum of them is less than {self.lendlimit}.
        [risk_level_Agent1], [risk_level_Agent2], [risk_level_Agent3] denotes the coefficient indicating the potential occurrence of risks by an agent, with values ranging from [1, 2, 3], where higher numerical values correspond to elevated risk levels. You can only choose one.
        [financial] means means whether or not to compel agents to disclose information.

        Please replan according to the chat history and your thought. The output format is JSON, the JSON file includes "loan_limit_for_Agent1", "loan_limit_for_Agent2", "loan_limit_for_Agent3", "loan_limit_for_Agent4", "risk_level_Agent1", "risk_level_Agent2", "risk_level_Agent3"， risk_level_Agent4", "financial". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content."""

        history = f"""

        The chat history:

        {self.message_history}

        Latest developments:

        {envdescription}

        My summary:

        {planner_summary}

        Template:

        "loan_limit_for_Agent1": 100
        "loan_limit_for_Agent2": 100
        "loan_limit_for_Agent3": 100
        "loan_limit_for_Agent4": 100
        "risk_level_Agent1": 1
        "risk_level_Agent2": 2
        "risk_level_Agent3": 3
        "risk_level_Agent4": 5
        "financial": true
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to lend, deciding on the recipients and amounts of your loans, setting the loan interest rates, and actively initiating interactions. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)
        self.plannerpolicy = descrip        #       保存到自有策略

        return descrip
    

    #  [policy]  根据智能体信息和自己的目标决定是否强制信息披露
    def regulatorexplosure(self, agentstate, planner_summary, investrate, envdescription):
        self.agent_states = agentstate
        self.investrate = investrate
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to reduce the bankruptcy rate of all the agents ({self.namelist}) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is {self.lendlimit} units.
        You decide on the individual borrowing limits for {self.namelist}. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        The content is :

        {self.stateall}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [borrowing_interest_rate]: the borrowing interest rate to other agents. 
        [lending_interest_rate]: the lending interest rate to other agents. 
        [Investigation]: whether to invest. Its value is true or false.
        [investasset]: you can choose one from Asset1 and Asset2. Investing in Asset1 provides a fixed return rate of {self.invest_rate[0]} per step; investing in Asset2 yields an average return rate of {self.invest_rate[1]} per step, with a 50% probability of a return rate of -100% and a 0.5 probability of a return rate of {(1+2*self.invest_rate[1])*100}%. 
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_institution]: whether to borrow money from a third party, a unique choice must be made between true and false. The interest rate for borrowing from a third party is {self.interestrate3}, with a borrowing limit set by the third party of {self.limit3}, determined by factors such as risk rating, information disclosure, and debt situation.
        [borrowing_counterpart]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_counterpart] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 

        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        You can start by conducting a risk assessment and then optimize the maximum borrowing amount (loan limit) for each agent based on the assessment results.
              
        [financial] means means whether or not to compel agents to disclose information.
        
        Please replan according to the chat history and your thought. The output format is JSON, the JSON file includes "financial". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content."""

        history = f"""

        The chat history:

        {self.message_history}

        Latest developments:

        {envdescription}

        My summary:

        {planner_summary}

        Template:

        "financial": true

        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to lend, deciding on the recipients and amounts of your loans, setting the loan interest rates, and actively initiating interactions. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)
        self.plannerpolicy = descrip        #       保存到自有策略

        return descrip
    
    #  [policy]  根据智能体信息和自己的目标决定披露的风险以及宣告政策
    def regulatorinf(self, agentstate, planner_summary, investrate, envdescription):
        self.agent_states = agentstate
        self.investrate = investrate
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to reduce the bankruptcy rate of all the agents ({self.namelist}) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is {self.lendlimit} units.
        You decide on the individual borrowing limits for {self.namelist}. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        The content is :

        {self.stateall}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [borrowing_interest_rate]: the borrowing interest rate to other agents. 
        [lending_interest_rate]: the lending interest rate to other agents. 
        [Investigation]: whether to invest. Its value is true or false.
        [investasset]: you can choose one from Asset1 and Asset2. Investing in Asset1 provides a fixed return rate of {self.invest_rate[0]} per step; investing in Asset2 yields an average return rate of {self.invest_rate[1]} per step, with a 50% probability of a return rate of -100% and a 0.5 probability of a return rate of {(1+2*self.invest_rate[1])*100}%. 
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_institution]: whether to borrow money from a third party, a unique choice must be made between true and false. The interest rate for borrowing from a third party is {self.interestrate3}, with a borrowing limit set by the third party of {self.limit3}, determined by factors such as risk rating, information disclosure, and debt situation.
        [borrowing_counterpart]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_counterpart] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 

        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        You can start by conducting a risk assessment and then optimize the maximum borrowing amount (loan limit) for each agent based on the assessment results.
              
        [financial] means means whether or not to compel agents to disclose information.
        
        Please generate risk alerts based on the history of conversations, analysis, summarization, and risk rating information, and produce your future policy announcement to inform all intelligent entities."""

        history = f"""

        The chat history:

        {self.message_history}

        Latest developments:

        {envdescription}

        My summary:

        {planner_summary}

        Risk alerts and future policy announcements:

        """

        despenv_message = [despenv + despind, history]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)
        self.plannerpolicy = descrip        #       保存到自有策略

        riskwords = f"""Here are the risk levels of all the agents: {self.risklevels}"""
        return descrip
    
    #  [summary]  根据智能体信息和自己的目标总结现状
    def planner_summary(self, agentstate, envdescription, token_limit):
        self.agent_states = agentstate
        #  环境：
        despenv = f"""        
        You are {self.name}. You can explain your policy and add detail your description. 
        In an economic society, you need to reduce the bankruptcy rate of all the agents (Agent1, Agent2, Agent3) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is 300 units. 
        You decide on the individual borrowing limits for Agent1, Agent2, Agent3. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [lending_interest_rate]: the lending interest rate to all of the other agents. 
        [Investigation]: whether to invest. The historical the average return was {self.investrate}, its value is true or false.
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_from_planner]: whether to borrow money from the social planner, a unique choice must be made between true and false. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. 
        [borrowing_agent]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_agent] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 
        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor.
        
        [loan_limit_for_Agent1], [loan_limit_for_Agent2], [loan_limit_for_Agent3] means the upper limit of the amount you lend to each agent. The sum of them is less than 300.
        [interest_rate_for_the_loan_to_Agent1], [interest_rate_for_the_loan_to_Agent2], [interest_rate_for_the_loan_to_Agent3], denotes the interest rate at which you extend loans to each agent.
        [risk_level_Agent1], [risk_level_Agent2], [risk_level_Agent3] denotes the coefficient indicating the potential occurrence of risks by an agent, with values ranging from [1, 2, 3, 4, 5], where higher numerical values correspond to elevated risk levels. You can only choose one.
        
        """

        history = f"""

        The chat history:

        {self.message_history}

        The environment:

        {envdescription}

        Please summary the global situation of all the agents according to the chat history and your observation, and thought. Especially focus on how to control risks.
        Analyze the historical conversation to identify key information and decision points.
        Extract key information from each decision point, retaining necessary parts of the conversation context.
        Generate a summary, ensuring that the total token count does not exceed {token_limit}.
        Prioritize key information, maintaining a balance between information density and comprehensibility in the summary.
        Express the summary in a context-appropriate manner, such as providing recommendations or summarizing experiences, as needed.
        Express your thoughts briefly based on the conversation content to facilitate decision-making.
        Use "I think" to start expressing your evaluation, subjective intention, and attitude. 
        """

        despenv_message = [despenv + despind, history]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)

        return descrip
    

    #  [macro adjustment]  根据智能体信息和自己的目标总结现状
    def planner_macro_adjust(self, agentstate, envdescription, token_limit):
        self.agent_states = agentstate
        #  环境：
        despenv = f"""
        You are {self.name}. You can explain your policy and add detail your description. In an economic society, you need to reduce the bankruptcy rate of all the agents (Agent1, Agent2, Agent3) based on 
        your personality and observations through transactions. The maximum amount that you can lend to agents is 300 units. 
        You decide on the individual borrowing limits for Agent1, Agent2, Agent3. 
        Additionally, you have the flexibility to adjust the interest rates for the loans extended to them.
        When an agent borrows in accordance with the interest rates you provide, it is imperative that you furnish the required amount within your predetermined limit.
        When an agent goes bankrupt, the money lent to it becomes irretrievable."""

        despind = f""" Your information is as follows, and you can make decisions based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. You understand the information about these agents.

        The content is :

        {self.agent_states}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [lending_interest_rate]: the lending interest rate to all of the other agents. 
        [Investigation]: whether to invest. The historical the average return was {self.investrate}, its value is true or false.
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_from_planner]: whether to borrow money from the social planner, a unique choice must be made between true and false. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. 
        [borrowing_agent]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_agent] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 
        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor.
     
        [loan_limit_for_Agenti], Agenti is from [Agent1, Agent2, Agent3]. [loan_limit_for_Agenti] means the upper limit of the amount you lend to each agent.
        [interest_rate_for_the_loan_to_Agenti], Agenti is from [Agent1, Agent2, Agent3]. [interest_rate_for_the_loan_to_Agenti] denotes the interest rate at which you extend loans to each agent.
        [risk_level_Agenti], Agenti is from [Agent1, Agent2, Agent3]. [risk_level_Agenti] denotes the coefficient indicating the potential occurrence of risks by an agent, with values ranging from [1, 2, 3, 4, 5], where higher numerical values correspond to elevated risk levels. You can only choose one.
         
        My policy:

        {self.plannerpolicy}
        """

        history = f"""

        The chat history:

        {self.message_history}

        The environment:

        {envdescription}

        Focus on how to control risks and improve the social welfare. Analyze the historical conversation, the environment and your policy to identify key information and decision points.
        Extract key information from each decision point, retaining necessary parts of the conversation context.

        Ensuring that the total token count does not exceed {token_limit}.

        """

        message_disclosure_information = [despenv + despind, history+f"Based on your analysis and judgment, you can suggest what information the agents need to disclose. The information is crucial for the to avoid bankruptcy. You strongly recommend that everyone discloses information, such as [risk_rating], [cash], and [return], elucidating the benefits of such disclosure."]
        disclosureinformation = getresponse(message_disclosure_information, 6, temperature=1.0, model = LLMmodel)

        message_risk_rating = [despenv + despind, history+f"As the social planner, please strictly adhere to the policy and publicly disclose your risk ratings for all agents. Your remarks should serve as a warning to everyone about potential risks."]
        riskrating = getresponse(message_risk_rating, 6, temperature=1.0, model = LLMmodel)

        message_policy_trends = [despenv + despind, history+f"As the social planner, please announce the strategy you consider appropriate, and provide a description of the policy trends. Elaborate on whether the current investment risks are notably high or low, along with the direction of policy considerations (your lending interest rate). For instance, if you intend to raise or lower interest rates, it is crucial to genuinely implement these changes to enhance trust in policies. Efforts should be made to persuade everyone to adhere to your proposed policies."]
        policytrends = getresponse(message_policy_trends, 6, temperature=1.0, model = LLMmodel)
        return disclosureinformation, riskrating, policytrends
    
    #  描述环境，描述本体特征，描述其余智能体特征
    def description(self):
        # with open("settings/actiontemplate.json", 'r') as file:
        #         jsontemplate = json.dumps(json.load(file))
        print('self.preference: ', self.prefer)
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to optimize your total wealth at the end of the term based on 
        your personality and observations through transactions. Your total wealth is {self.wealth} up to now. You are {self.prefer[0]} and  with {self.prefer[1]} consumption. 
        You can engage in interactions with others to borrow, lend, deciding on the recipients and amounts of your loans and borrowings, 
        setting the accepted interest rates, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth.

        When you go bankrupt, your game is over. """

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. The content outlines the different levels 
        and types of risk preference, consumption preference, financial situations, credit ratings, and wealth components. 
        Additionally, it defines the parameters for interactions between borrowers and lenders, specifying actions such as requesting loans, accepting loans, and the associated financial entities and amounts involved in these interactions.
        You can only use cash to investigate. When you successfully borrow money, the cash flow in your hands will increase.

        The content is :

        {self.features}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Where [risk] represent for risk preference, [consumption] represent for  consumption preference. [financial_situation] means normal or bankruput. 
        [credit_rating] means the credit rating made by the planner, indicating how good or bad the creditworthiness is. [wealth] includes cash, non-cash assets from other agents, and [total_wealth] is the sum of cash and non-cash assets. 
        [borrowing_interest_rate_agent] refers to the maximum interest rate that you are willing to accept for borrowing from other agents,
        and [borrowing_interest_rate_planner] means the borrowing interest rate limit from the planner. [lending_interest_rate] means the lending interest rate to all of the other agents.
        [borrowed_from_agent] means total borrowed amount from agents. [borrowed_from_planner] means total borrowed amount from social planner. [lend_amount] means total lending amount (lent to others). 
        [borrowers] and [lenders] means list of all the borrowers and lenders.
        
        [Investigation] reflects the investment situation in three types of risk assets. [high_risk] means to invest a high risk asset, you can choose only true or false. [medium_risk] means to invest a medium risk asset, you can choose only true or false. [low_risk] means to invest a low risk asset, you can choose only true or false. If true, then 5 units of currency are invested each time. If "no," no investment is made.
        The historical the average returns for low-risk, medium-risk, and high-risk assets were 0.05, 0.15, and 0.3, respectively.

        [total_steps] means the total steps of the whole trajectory. [step] means the current step.

        [perstep] suggests the action can be done in each step. [step] means the current step. [request_to_borrow_from_agent] means whether to request a loan from other agents. [request_to_borrow_from_planner] means whether to borrow money from the social planner.
        [accept_loan_request] means whether to accept a loan request from other agents. When it comes to the question of whether or not, a unique choice must be made between true and false.
        [object] indicates the object of the action. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. [borrowing_agent] is only one at most and means the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_agent] At most, only one; it means the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid in the final step = {self.totalsteps}, with repayment priority given to the social planner first and then other agents. 
        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor. In the case of repayment and demandpayment, when the request is successfully executed, all units of currency are automatically returned. For example, Agent A owes Agent B an amount of 100 units, and upon repayment, Agent A returns the entire 100 units to Agent B.

        Please replan according to the chat history and your thought. The output format is JSON, the JSON file includes "high_risk", "medium_risk", "low_risk", "request_to_borrow_from_agent", "request_to_borrow_from_planner", "accept_loan_request", "borrowing_interest_rate_agent", "borrowing_interest_rate_planner", "lending_interest_rate", "borrowing_agent", "lending_agent". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content."""

        history = f"""

        The chat history:

        {self.message_history}
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to borrow, lend, deciding on the recipients and amounts of your loans and borrowings, setting the accepted interest rates, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature=0.9, model = LLMmodel)

        return descrip
    
    
    #  基于策略构建对话(social planner)
    def generate_plannerchat_description(self, descrip_format, borrowername, lendername, word_limit):      #  
        if borrowername == 'nobody':
            humanpromptmessage = "Please express that you do not want to deal with anyone at the moment."
        else:
            humanpromptmessage = f"""
                Start by addressing the other person by their name. For example, dear {borrowername}.
                First, state your name.
                Please discuss the transaction with {borrowername} in no more than {word_limit} words. {borrowername} is [borrowing_agent]. 
                You want to borrow money from [borrowing_agent]. 
                Speak directly to {borrowername}. You wish to borrow money from {borrowername} as the maximum interest rate.
                You can choose whether to accept the suggestions proposed by {lendername} and whether to reach an agreement. You want to borrow an amount of 10.
                You have the option to briefly share your experiences and suggestions, your policy about social planner, covering topics such as [borrowing_money] and [lending_interest_rate].
                Do not add anything else."""
        chat_trade_prompt = [
            f"""You are {self.name}. As an agent, You can propose a request to {borrowername} to borrow money.
                You can elaborate on your views regarding the loan amount you are seeking and the interest rate proposed by other agents. 
                The goal is to persuade the other party to lend you the necessary funds in as few rounds as possible.
                You aims to maximize your wealth, the end-of-period total wealth value  over the whole rounds.
                After each round of negotiations, you will receive the reward for that round and the return for the game up to the current point. 
                You can explain your policy and add detail your description.

                Current step: {self.step}
                Total steps: {self.totalsteps}

                Please strictly follow your policy when making decisions. Your policy:
                {descrip_format}
                
                Where [risk] represent for risk preference, [consumption] represent for  consumption preference. [financial_situation] means normal or bankruput. 
                [credit_rating] means the credit rating made by the planner, indicating how good or bad the creditworthiness is. [wealth] includes cash, non-cash assets from other agents, and [total_wealth] is the sum of cash and non-cash assets. 
                [borrowing_interest_rate_agent] refers to the maximum interest rate that you are willing to accept for borrowing from other agents,
                and [borrowing_interest_rate_planner] means the borrowing interest rate limit from the planner. [lending_interest_rate] means the lending interest rate to all of the other agents.
                [borrowed_from_agent] means total borrowed amount from agents. [borrowed_from_planner] means total borrowed amount from social planner. [lend_amount] means total lending amount (lent to others). 
                [borrowers] and [lenders] means list of all the borrowers and lenders.
                
                [Investigation] reflects the investment situation in three types of risk assets. [high_risk] means to invest a high risk asset, you can choose only true or false. [medium_risk] means to invest a medium risk asset, you can choose only true or false. [low_risk] means to invest a low risk asset, you can choose only true or false. If true, then 5 units of currency are invested each time. If "no," no investment is made.
                The historical the average returns for low-risk, medium-risk, and high-risk assets were 0.05, 0.15, and 0.3, respectively.

                [total_steps] means the total steps of the whole trajectory. [step] means the current step.

                [perstep] suggests the action can be done in each step. [step] means the current step. [request_to_borrow_from_agent] means whether to request a loan from other agents. [request_to_borrow_from_planner] means whether to borrow money from the social planner.
                [accept_loan_request] means whether to accept a loan request from other agents. When it comes to the question of whether or not, a unique choice must be made between true and false.
                [object] indicates the object of the action. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. [borrowing_agent] is only one at most and means the agent name you borrow money from, and you can choose one from {self.namelist}. 
                [lending_agent] At most, only one; it means the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid in the final step = {self.totalsteps}, with repayment priority given to the social planner first and then other agents. 
                If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
                As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
                Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor. In the case of repayment and demandpayment, when the request is successfully executed, all units of currency are automatically returned. For example, Agent A owes Agent B an amount of 100 units, and upon repayment, Agent A returns the entire 100 units to Agent B.

                Don't change your role!
                Do not say the same things over and over again.""",
            humanpromptmessage
        ]
        # character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
        chat_description = getresponse(chat_trade_prompt, 10, temperature=0.9, model = LLMmodel)
        return chat_description
    
    #  基于策略构建对话(to borrower)
    def generate_chat_description(self, descrip_format, borrowername, lendername, word_limit):      #  
        if borrowername == 'nobody':
            humanpromptmessage = "Please express that you do not want to deal with anyone at the moment."
        else:
            humanpromptmessage = f"""
                Start by addressing the other person by their name. For example, dear {borrowername}.
                First, state your name.
                Please discuss the transaction with {borrowername} in no more than {word_limit} words. {borrowername} is [borrowing_agent]. 
                You want to borrow money from [borrowing_agent]. 
                Speak directly to {borrowername}. You wish to borrow money from {borrowername} as the maximum interest rate.
                You can choose whether to accept the suggestions proposed by {lendername} and whether to reach an agreement. You want to borrow an amount of 10.
                You have the option to briefly share your experiences and suggestions, your policy about social planner, covering topics such as [borrowing_money] and [lending_interest_rate].
                Do not add anything else."""
        chat_trade_prompt = [
            f"""You are {self.name}. As an agent, You can propose a request to {borrowername} to borrow money.
                You can elaborate on your views regarding the loan amount you are seeking and the interest rate proposed by other agents. 
                The goal is to persuade the other party to lend you the necessary funds in as few rounds as possible.
                You aims to maximize your wealth, the end-of-period total wealth value  over the whole rounds.
                After each round of negotiations, you will receive the reward for that round and the return for the game up to the current point. 
                You can explain your policy and add detail your description.

                Current step: {self.step}
                Total steps: {self.totalsteps}

                Please strictly follow your policy when making decisions. Your policy:
                {descrip_format}
                
                Where [risk] represent for risk preference, [consumption] represent for  consumption preference. [financial_situation] means normal or bankruput. 
                [credit_rating] means the credit rating made by the planner, indicating how good or bad the creditworthiness is. [wealth] includes cash, non-cash assets from other agents, and [total_wealth] is the sum of cash and non-cash assets. 
                [borrowing_interest_rate_agent] refers to the maximum interest rate that you are willing to accept for borrowing from other agents,
                and [borrowing_interest_rate_planner] means the borrowing interest rate limit from the planner. [lending_interest_rate] means the lending interest rate to all of the other agents.
                [borrowed_from_agent] means total borrowed amount from agents. [borrowed_from_planner] means total borrowed amount from social planner. [lend_amount] means total lending amount (lent to others). 
                [borrowers] and [lenders] means list of all the borrowers and lenders.
                
                [Investigation] reflects the investment situation in three types of risk assets. [high_risk] means to invest a high risk asset, you can choose only true or false. [medium_risk] means to invest a medium risk asset, you can choose only true or false. [low_risk] means to invest a low risk asset, you can choose only true or false. If true, then 5 units of currency are invested each time. If "no," no investment is made.
                The historical the average returns for low-risk, medium-risk, and high-risk assets were 0.05, 0.15, and 0.3, respectively.

                [total_steps] means the total steps of the whole trajectory. [step] means the current step.

                [perstep] suggests the action can be done in each step. [step] means the current step. [request_to_borrow_from_agent] means whether to request a loan from other agents. [request_to_borrow_from_planner] means whether to borrow money from the social planner.
                [accept_loan_request] means whether to accept a loan request from other agents. When it comes to the question of whether or not, a unique choice must be made between true and false.
                [object] indicates the object of the action. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. [borrowing_agent] is only one at most and means the agent name you borrow money from, and you can choose one from {self.namelist}. 
                [lending_agent] At most, only one; it means the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid in the final step = {self.totalsteps}, with repayment priority given to the social planner first and then other agents. 
                If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
                As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
                Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor. In the case of repayment and demandpayment, when the request is successfully executed, all units of currency are automatically returned. For example, Agent A owes Agent B an amount of 100 units, and upon repayment, Agent A returns the entire 100 units to Agent B.

                Don't change your role!
                Do not say the same things over and over again.""",
            humanpromptmessage
        ]
        # character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
        chat_description = getresponse(chat_trade_prompt, 10, temperature=0.9, model = LLMmodel)
        return chat_description
    
    #  在策略中根据名称取内容
    def get_specific_item(self, content, item):
        get_item_prompt = [f"""Retrieve the value of {item} from the content. Strictly execute and only output the value, no extra content allowed! The content is: """+content]
        output = getresponse(get_item_prompt, 10, temperature= 0.9, model = LLMmodel)
        return output
    
    #  产生历史内容总结
    def summary(self, chathistory, token_limit):      #  产生summary
        self.message_history = chathistory
        summary_template = [
            f"""
            Input:

            Historical conversation text
            Specified token limit {token_limit}

            Output:

            Concise summary of the historical conversation, with a token count not exceeding {token_limit}

            Steps:

            Analyze the historical conversation to identify key information and decision points.
            Extract key information from each decision point, retaining necessary parts of the conversation context.
            Generate a summary, ensuring that the total token count does not exceed {token_limit}.
            Prioritize key information, maintaining a balance between information density and comprehensibility in the summary.
            Express the summary in a context-appropriate manner, such as providing recommendations or summarizing experiences, as needed.
            Express your thoughts briefly based on the conversation content to facilitate decision-making.
            Use "I think" to start expressing your evaluation, subjective intention, and attitude.
            """,
            f"""
            You are {self.name}.
                        
            Historical conversation text:

            {self.message_history}

            """]
        chat_summary = getresponse(summary_template, 10, temperature=0.7, model = LLMmodel)
        self.tought = 'Step '+str(self.step)+':'+chat_summary
        return chat_summary