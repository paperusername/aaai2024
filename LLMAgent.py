#  能顺利运行，增加了动态状态转移，环境返回信息
from openai import OpenAI
import re
import copy
import os
import numpy as np
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

class AdaptiveAgent():
    def __init__(self, input_size, output_size, features):
        super(AdaptiveAgent, self).__init__()
        
        #  定义特征
        self.feature_agent_i = features[0]
        # print('features: ', features)
        self.name = features[0]['name']
        self.namelist = features[0]['namelist']
        self.numagent = len(self.namelist)      #  智能体数量
        self.prefer = features[0]['preference']['risk']
        self.wealth = features[0]['wealth']['total_wealth']
        self.cash = features[0]['wealth']['cash']
        self.totalsteps = features[0]['interaction']['total_steps']
        self.step = features[0]['interaction']['step']
        self.financial_situation = "normal"
        self.investigation = True
        self.investasset = "Asset1"

        #  动态变化的量
        self.balance = 0        #  余额 ==================================================================================================
        self.borrowingmoney = 0     #  全部的借款（可变）====================================================================================

        self.thought = ''   #  定义总结语句库
        self.policy_var_data = {}       #  存放智能体策略
        self.state = copy.deepcopy(self.feature_agent_i)     #  定义初始观测空间
        self.borrow_rate_all = [[] for _ in range(self.numagent)]       #  记录所有借入单的利率
        self.borrow_rate_planner = []       #  记录所有借入计划者单的利率
        self.num_borrow_planner = 0     #  向计划者借钱的次数
        self.lend_rate_all = [[] for _ in range(self.numagent)]         #  记录所有借出金额的利率
        self.borrow_step_all = [[] for _ in range(self.numagent)]       #  记录所有借入单的时间步
        self.borrow_step_planner = []       #  记录借入计划者单的时间步
        self.lend_step_all = [[] for _ in range(self.numagent)]         #  记录所有借出单的时间步
        self.borrow_amount = 0      #  记录所有借入金额总和
        self.lend_amount = 0        #  记录所有借出金额总和

        #  定义默认参数
        self.invest_rate = [0.05, 0.1]    #  投资回报率【固定收益、风险平均收益】==============================================================================
        self.amount_per_invest = 20      #  每次投的金额====================================================================================
        self.unitcash = 10      #  单个时间步的资金流通单位

        self.return_step = 0      #  投资周期总收益
        self.return_borrow_step_agent = 0    #  从agent借款金额变化
        self.return_borrow_step_planner = 0     #  从planner处借款收益变化
        self.return_borrow_step =  0        #  共计借款收益变化
        self.return_lend_step = 0       #  共计借出款收益变化
        self.lendrate = 0   #  向外借款利率
        self.interestrate3 = 0.08      #  第三方设定的借款利率（固定）=================================================================================================
        self.limit3 = 0     #  第三方设定的借款金额上限（因人而异，结构化政策）=======================================================================================
        

        #  智能体互动矩阵
        self.agentmat = []
        #  social planner传递消息矩阵
        self.plannermat = []
        #  social planner所有公示消息   ===============重要！！！！！！！！！！=======================================================================================================================
        self.info_social_planner = []

        # self.system_message = system_message        #  定义系统消息
        # self.system_message_format = system_message_format

        self.prefix = f"{self.name}: "
        self.reset()
        self.summarize = ["This is the summay."]     # 定义总结语言
    
    def reset(self):        #  初始化：智能体对话历史，环境的描述
        self.message_history = ["Here is the conversation so far."]
        self.envdescription = [""]      #  初始化环境描述
        self.num_borrow_planner = 0 

    def send(self, chathistory, socialplannermsg, word_limit) -> str:       #  输入：对话历史，监管者消息，输出：策略，借款对话
        self.info_social_planner = socialplannermsg     #  赋值social planner披露信息
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        self.message_history = chathistory
        #  生成借智能体策略 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++补充环境观察
        while True:
            try:
                descrip_format  = self.decision()
                # print('Policy: ', descrip_format)
                #  生成借款对话
                # borrow_name = self.get_specific_item(descrip_format, item = "[borrowing_agent]")
                descrip_format_deal1 = re.split("{", descrip_format)[1]
                descrip_format_deal2 = re.split("}", descrip_format_deal1)[0]
                descrip_format_deal = "{"+descrip_format_deal2+"}"
                # print("descrip_format_deal: ", descrip_format_deal)
                descrip_format_data = json.loads(descrip_format_deal)
                self.policy_var_data = copy.deepcopy(descrip_format_data)       #  保存策略新生成的变量
                #  更新环境中的变量
                print('descrip_format_data: ', descrip_format_data)
                borrow_name = descrip_format_data['borrowing_counterpart']
                lender_name = descrip_format_data['lending_counterpart']
                self.investigation = descrip_format_data['investigation']
                self.investasset = descrip_format_data['investasset']
                # print(borrow_name)
                break
            except json.JSONDecodeError as e:
                NotImplementedError
                # print(f"JSON decoding error: {e}")
        # print('borrow_name: ', borrow_name)
        borrower_chat = self.generate_chat_description(descrip_format, borrow_name, lender_name, word_limit)
        print('Chat to borrower: ', borrower_chat)

        return descrip_format_deal, borrower_chat   #  policy message and chatmessage separately

    def receive(self, name: str, message: str) -> None:     #  收到说话人消息
 
        """
        Concatenates {message} spoken by {name} into message history
        """
 
        self.message_history.append(f"{name}: {message}")

    def receive_load_save(self, name: str, message: str) -> None:       #  从库中加载和保存
 
        """
        Concatenates {message} spoken by {name} into message history
        """
 
        self.message_history.append(f"{name}: {message}")

        # save json file
        chatsavepath = "data/History/ChatHistory/"       #   增加文件名变量==============================================================================================================
        output_file_path = chatsavepath + "chatlog.json"
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            json.dump(self.message_history, output_file, ensure_ascii=False, indent=2)
    
    def update(self, currentstep, Agent_match_matrix, Agent_financial_situation, Plannermat):       #  智能体更新状态[当前时间步, 其余智能体利率, 智能体匹配矩阵, 计划者矩阵]
        self.step = currentstep     #  步数（决策后更新）
        env_feedback_descrip = []     #  环境反馈语言

        self.agentmat = Agent_match_matrix      #  更新和赋值
        self.plannermat = Plannermat    

        #  更新风险评级
        self.state["credit_rating"] = copy.deepcopy(Plannermat[self.name]["risk_level_"+self.name])

        #  破产则欠款清零
        env_financial_descrip = f" "        #  反馈1==================================================================
        numbankrupt = 0
        for i in range(len(Agent_financial_situation)):
            if Agent_financial_situation[i] == 'bankrupt':
                numbankrupt += 1
                self.lend_rate_all[i] = []
                env_financial_descrip = env_financial_descrip + f"""{self.name} is bankrupt, unable to repay the debt, so all the money {self.name} owes to others has been cleared."""
        if numbankrupt == 0:
            env_financial_descrip = f"""All agents' finances are in good order."""
        else: 
            env_financial_descrip = f"""Number of bankrupt agents: {numbankrupt}.""" + env_financial_descrip

        # print('state: ', self.state)
        # print('policy: ', self.policy_var_data)

        #  借款和还款
        #  whether to borrow or lend and the object
        self.state['interaction']['object']['request_to_borrow_from_agent'] = (self.policy_var_data['borrowing_counterpart'] != 'nobody')

        #  the object
        self.state['interaction']['object']['borrowing_counterpart'] = copy.deepcopy(self.policy_var_data['borrowing_counterpart'])
        self.state['interaction']['object']['lending_counterpart'] = copy.deepcopy(self.policy_var_data['lending_counterpart'])
        
        #  interest rate
        self.state['wealth']['lending_interest_rate'] = copy.deepcopy(self.policy_var_data['lending_interest_rate'])

        print("self.state: ", self.state)

        #  计算借入款金额（智能体）
        borrow_money_list = [0 for _ in range(self.numagent)]
        for i in range(self.numagent):
            if len(self.borrow_rate_all[i]) != 0:
                borrow_money_i = np.sum(np.array(self.borrow_rate_all[i]))*self.unitcash
                borrow_money_list[i] = borrow_money_i
            else:
                borrow_money_list[i] = 0
        self.return_borrow_step_agent = np.sum(np.array(borrow_money_list))       #  计算借款损失=========================================================

        #  计算借入款金额（从计划者处）
        self.return_borrow_step_planner = self.borrow_rate_planner*self.num_borrow_planner*self.unitcash

        #  计算借款损失=========================================================
        self.return_borrow_step = self.return_borrow_step_agent + self.return_borrow_step_planner       #  计算借款损失总和

        #  计算借出款金额 =====================================================================================================
        lend_money_list = [0 for _ in range(self.numagent)]
        for i in range(self.numagent):
            if len(self.lend_rate_all[i]) != 0:
                lend_money_i = np.sum(np.array(self.lend_rate_all[i]))*self.unitcash
                lend_money_list[i] = lend_money_i
            else:
                lend_money_list[i] = 0
        self.return_lend_step = np.sum(np.array(lend_money_list))

        env_borrow_descrip = f""" """       #  反馈2==========================================
        sign_agent_borrow  = 0      #  向Agent借款的标志
        #  从Agent处借款；记录每一次借入款的interest rate
        if self.policy_var_data['borrowing_counterpart'] != 'nobody':
            agenti = self.policy_var_data['borrowing_counterpart']        #  命名coincidence agent
            if Agent_match_matrix[agenti]['lender'] == self.name:
                sign_agent_borrow = 1
                env_borrow_descrip = f"""{self.name} successfully borrowed {self.unitcash} units of cash from {agenti} at an interest rate of {Agent_match_matrix[agenti]['lend_rate']}."""
                self.state['wealth']['cash'] = copy.deepcopy(self.state['wealth']['cash']) + self.unitcash
                if self.policy_var_data['borrowing_counterpart'] == 'Agent1':
                    self.lend_rate_all[0].append(Agent_match_matrix[agenti]['lend_rate'])
                if self.policy_var_data['borrowing_counterpart'] == 'Agent2':
                    self.lend_rate_all[1].append(Agent_match_matrix[agenti]['lend_rate'])
                if self.policy_var_data['borrowing_counterpart'] == 'Agent3':
                    self.lend_rate_all[2].append(Agent_match_matrix[agenti]['lend_rate'])
                if self.policy_var_data['borrowing_counterpart'] == 'Agent4':
                    self.lend_rate_all[3].append(Agent_match_matrix[agenti]['lend_rate'])
            else:
                env_borrow_descrip = f""" Step = {self.step}, {self.name} did not borrow any money."""

        #  更新持有现金
        #  从个体处借款
        #  从regulator 处借款
        sign_regulator_borrow = 0       #  向regulator借款标志
        if  self.policy_var_data['request_to_borrow_from_institution'] == True:
            sign_regulator_borrow = 1
            self.state['wealth']['cash'] = copy.deepcopy(self.state['wealth']['cash']) + self.unitcash

        env_lend_descrip = f""" """     #  反馈4==========================================
        sign_agent_lend = 0     #  借款给agent标志
        #  记录每一次借出款的interest rate===========================================================================
        if self.policy_var_data['lending_counterpart'] != 'nobody':
            agenti = self.policy_var_data['lending_counterpart']        #  命名coincidence agent
            if Agent_match_matrix[agenti]['borrower'] == agenti:
                sign_agent_lend = 1
                env_lend_descrip = f"""{self.name} successfully lent {self.unitcash} units of cash to {agenti} at an interest rate of {Agent_match_matrix[self.name]['lend_rate']}."""
                self.state['wealth']['cash'] = copy.deepcopy(self.state['wealth']['cash']) - self.unitcash
                if self.policy_var_data['lending_counterpart'] == 'Agent1':
                    self.lend_rate_all[0].append(Agent_match_matrix[self.name]['lend_rate'])
                if self.policy_var_data['lending_counterpart'] == 'Agent2':
                    self.lend_rate_all[1].append(Agent_match_matrix[self.name]['lend_rate'])
                if self.policy_var_data['lending_counterpart'] == 'Agent3':
                    self.lend_rate_all[2].append(Agent_match_matrix[self.name]['lend_rate'])
                if self.policy_var_data['lending_counterpart'] == 'Agent4':
                    self.lend_rate_all[2].append(Agent_match_matrix[self.name]['lend_rate'])
            else:
                env_lend_descrip = f""" Step = {self.step}, {self.name} did not lend any money."""

        #  投资
        self.state["investall"] = copy.deepcopy(self.state['wealth']['cash'])
        
        #  判断资产类型，进行收益计算
        if self.state["investasset"] == "Asset1":
            self.return_step = (1 + self.invest_rate[0])*copy.deepcopy(self.state["investall"])
            returnrate = self.invest_rate[0]
        else:
            self.return_step = (np.random.rand(1)<0.5)*(2 + 2*self.invest_rate[1])*copy.deepcopy(self.state["investall"])
            returnrate = self.invest_rate[1]

        env_investigation_descrip = f""" From last step to step = {self.step}, the return rate of investigation is {returnrate}. {self.name} get total return {self.return_step - self.state['wealth']['cash']}."""       #  反馈2==========================================

        self.state['wealth']['cash'] = self.return_step      #  更新投资后cash===================================================

        #  计算清算后的现金流
        self.state['wealth']['cash'] = copy.deepcopy(self.state['wealth']['cash']) - self.return_borrow_step + self.return_lend_step + (sign_agent_lend - sign_agent_borrow - sign_regulator_borrow) * self.unitcash

        #  计算总财富======================================================================================================================
        if self.state['wealth']['cash'] <= 0:
                self.state['financial_situation'] = 'bankrupt'      #  现金流不足就算破产
        #  更新破产状态(破产，所有欠款清零)==================================================================================================
        self.state['wealth']['total_wealth'] = copy.deepcopy(self.state['wealth']['total_wealth']) + self.return_step + self.return_borrow_step + self.return_lend_step        #   赚钱减去还钱========================================================
        self.wealth = copy.deepcopy(self.state['wealth']['total_wealth'])
        if self.state['wealth']['total_wealth'] < 0:
            self.financial_situation = "bankrupt"       #  登记破产，智能体失去决策权

        env_feedback_descrip.append(env_financial_descrip)
        env_feedback_descrip.append(env_investigation_descrip)
        env_feedback_descrip.append(env_borrow_descrip)
        env_feedback_descrip.append(env_lend_descrip)

        self.envdescription = env_feedback_descrip

        return env_feedback_descrip     #  返回环境描述

        
    #  描述环境，描述本体特征，描述其余智能体特征
    def description(self):
        # with open("settings/actiontemplate.json", 'r') as file:
        #         jsontemplate = json.dumps(json.load(file))
        print('self.preference: ', self.prefer)
        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to optimize your total wealth at the end of the term based on 
        your personality and observations through transactions. Your total wealth is {self.wealth} up to now. You are {self.prefer[0]}. 
        You can engage in interactions with others to borrow, lend, deciding on the recipients and amounts of your loans and borrowings, 
        setting the accepted interest rates, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth.

        When you go bankrupt, your game is over. """

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. You can only use cash to investigate. When you successfully borrow money, the cash flow in your hands will increase. You can get more cash by borrowing money.

        The content is :

        {self.state}

        Current step: {self.step}
        Total steps: {self.totalsteps}

        Explaination: 
        [risk] : risk preference. 
        [financial_situation]: normal or bankruput. 
        [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
        [total_wealth]: the sum of your assets. 
        [lending_interest_rate]: the lending interest rate to all of the other agents. 
        [Investigation]: whether to invest. The historical the average return was {self.invest_rate}, its value is true or false.
        [total_steps]: the total steps of the whole trajectory. 
        [step]: the current step.
        [request_to_borrow_from_planner]: whether to borrow money from the social planner, a unique choice must be made between true and false. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. 
        [borrowing_counterpart]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
        [lending_counterpart] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 
        If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor.

        Please replan according to the chat history and your thought. Please decide the output. The output format is JSON, the JSON file includes "invesitgation", "request_to_borrow_from_planner", "lending_interest_rate", "borrowing_counterpart", "lending_counterpart". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content.
        
        The template is:
        "investigation": true,
        "request_to_borrow_from_planner": false,
        "lending_interest_rate": 0.1,
        "borrowing_counterpart": "Agent2",
        "lending_counterpart": "nobody"
        
        """

        history = f"""

        The social planner's decision:

        {self.plannermat}

        The chat history:

        {self.message_history[-1]}

        The summary:

        {self.summarize}

        Public disclosure by the social planner:

        {self.info_social_planner}

        Action execution result last step:

        {self.envdescription}
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to borrow, lend, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature = 0, model = LLMmodel)
        # print('descrip: ', descrip)
        return descrip
    
    #  根据所有获得环境和智能体信息，生成决策
    def decision(self):

        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to optimize your total wealth at the end of the term based on 
        your personality and observations through transactions. Your total wealth is {self.wealth} up to now. You are {self.prefer[0]}. 
        You can engage in interactions with others to borrow, lend, deciding on the recipients and amounts of your loans and borrowings, 
        setting the accepted interest rates, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth.

        When you go bankrupt, your game is over. """

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. You can only use cash to investigate. When you successfully borrow money, the cash flow in your hands will increase. You can get more cash by borrowing money.

        The content is :

        {self.state}

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
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        Please replan according to the chat history and your thought. Please decide the output. The output format is JSON, the JSON file includes "invesitgation", "request_to_borrow_from_planner", "lending_interest_rate", "borrowing_counterpart", "lending_counterpart". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content.
        
        The template is:
        "request_to_borrow_from_institution": false,
        "lending_interest_rate": 0.1,
        "borrowing_interest_rate": 0.1,
        "borrowing_counterpart": "Agent2",
        "lending_counterpart": "nobody"
        "investigation": true,
        "investasset": "Asset1"
        
        """

        history = f"""

        The regulator's decision:

        {self.plannermat}

        The chat history:

        {self.message_history[-1]}

        The summary:

        {self.summarize}

        Public disclosure by the regulator:

        {self.info_social_planner}

        Action execution result last step:

        {self.envdescription}
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to borrow, lend, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature = 0, model = LLMmodel)
        # print('descrip: ', descrip)
        return descrip
    
    #  根据所有获得环境和智能体信息，生成投资决策
    def investdec(self):

        #  环境：
        despenv = f""" 
        Generate a JSON file based on the prompt words, without generating anything else.
        
        You are {self.name}. In an economic society, you need to optimize your total wealth at the end of the term based on 
        your personality and observations through transactions. Your total wealth is {self.wealth} up to now. You are {self.prefer[0]}. 
        You can engage in interactions with others to borrow, lend, deciding on the recipients and amounts of your loans and borrowings, 
        setting the accepted interest rates, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth.

        When you go bankrupt, your game is over. """

        despind = f""" Your information is as follows, and you can make decisions (only json file) based on the content below. You can only use cash to investigate. When you successfully borrow money, the cash flow in your hands will increase. You can get more cash by borrowing money.

        The content is :

        {self.state}

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
        As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
        Once the borrowing transaction is successfully executed, {self.unitcash} units of currency are transferred from the creditor to the debtor.

        Please replan according to the chat history and your thought. Please decide the output. The output format is JSON, the JSON file includes "invesitgation", "request_to_borrow_from_planner", "lending_interest_rate", "borrowing_counterpart", "lending_counterpart". 
        This is a format that needs to be strictly adhered to, referencing the form of its key rather than the content.
        
        The template is:
        "investigation": true,
        "investasset": "Asset1"
        
        """

        history = f"""

        The regulator's decision:

        {self.plannermat}

        The chat history:

        {self.message_history[-1]}

        The summary:

        {self.summarize}

        Public disclosure by the regulator:

        {self.info_social_planner}

        Action execution result last step:

        {self.envdescription}
        """

        despenv_message = [despenv + despind, history+f"You can engage in interactions with others to borrow, lend, and actively initiating interactions or accepting requests from others. You can use borrowed money to invest and accumulate wealth. Please generate the json file only."]
        descrip = getresponse(despenv_message, 6, temperature = 0, model = LLMmodel)
        # print('descrip: ', descrip)
        return descrip
    
    #  根据所有获得环境和智能体信息，进行信息披露
    def explosure(self):

        #  财务：
        despfinancial = f"""My balance stands at {self.balance}, with a borrowing amount of {self.borrowingmoney}, and {self.cash} in cash reserves. """
        
        return despfinancial
    
    #  基于策略构建对话(to borrower)
    def generate_chat_description(self, descrip_format, borrowername, lendername, word_limit):      #  
        if borrowername == 'nobody':
            humanpromptmessage = f"Please express that you do not want to deal with anyone at the moment. Please discuss the transaction in no more than {word_limit} words."
        else:
            humanpromptmessage = f"""
                Start by addressing the other person by their name. For example, dear {borrowername}.
                First, state your name.
                Please discuss the transaction with {borrowername} in no more than {word_limit} words. You can choose whether to accept the suggestions proposed by {lendername} and whether to reach an agreement in your lend interest rate {self.lendrate}.
                If you have successfully borrowed or repaid loans before, please consider sharing your experience with everyone.
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
                
                Explaination: 
                [risk] : risk preference. 
                [financial_situation]: normal or bankruput. 
                [credit_rating] : the credit rating made by the planner, indicating how good the creditworthiness is. 
                [total_wealth]: the sum of your assets. 
                [lending_interest_rate]: the lending interest rate to all of the other agents. 
                [Investigation]: whether to invest. The historical the average return was {self.invest_rate}, its value is true or false.
                [total_steps]: the total steps of the whole trajectory. 
                [step]: the current step.
                [request_to_borrow_from_planner]: whether to borrow money from the social planner, a unique choice must be made between true and false. You can borrow money from other agents and the planner. If you borrow money successfully, your cash will increase. If you lend money to other agents, your cash will decrease. 
                [borrowing_agent]: the agent name you borrow money from, and you can choose one from {self.namelist}. 
                [lending_agent] the agent name you lend money to, and you can choose one from {self.namelist}. All debts will be automatically repaid 1 step later. 
                If the agent does not go bankrupt, all the money will be repaid. If the agent goes bankrupt, the remaining amount will be returned.
                As for the agent, you can choose one from {self.namelist}. If you do not want to choose any agent in the list, please output "nobody".
                Once the borrowing transaction is successfully executed, 10 units of currency are transferred from the creditor to the debtor.
                
                The social planner's decision:
                {self.plannermat}
        
                The chat history:

                {self.message_history[-1]}

                Public disclosure by the social planner:
                {self.info_social_planner}

                Action execution result last step:

                {self.envdescription}

                The summary:

                {self.summarize}

                Don't change your role!
                Do not say the same things over and over again.""",
            humanpromptmessage
        ]
        # character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
        chat_description = getresponse(chat_trade_prompt, 10, temperature=0.9, model = LLMmodel)
        return chat_description
    
    #  在策略中根据名称取内容
    def get_specific_item(self, content, item):
        get_item_prompt = [f"""Retrieve the value of {item} from the content. Strictly execute and only output the value, no extra content allowed!""", """The content is: """+content]
        output = getresponse(get_item_prompt, 10, temperature= 0.9, model = LLMmodel)
        return output
    
    #  产生历史内容总结
    def summary(self, chathistory, token_limit):      #  产生summary
        self.message_history = chathistory
        summary_template = [
            f"""
            You are {self.name}.
            Input:

            Historical conversation text

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
                         
            The regulator's decision:

            {self.plannermat}

            Historical conversation text:

            {self.message_history}

            Your summary:

            """]
        chat_summary = getresponse(summary_template, 10, temperature=0.7, model = LLMmodel)
        self.tought = 'Step '+str(self.step)+':'+chat_summary
        self.summarize = chat_summary       #  更新summary
        return chat_summary
    
    # def summary(self, history):     #   修改===============================================================================================================
    #     summary_of_state = f"""I am {self.name}. This is my summary: """
    #     return summary_of_state