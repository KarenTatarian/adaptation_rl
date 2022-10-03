'''
reward function:
R = = - alpha*dh + gamma*(sqrt(tobj/tdecide)) + beta* decision - delta*degree
'''
import numpy as np
import random

class Reward_Function:

    def __init__(self, behavior_array=[None,None,None,None]):
        self.behavior_array = behavior_array
        self.prox = behavior_array[0]
        self.gest = behavior_array[1]
        self.gaze = behavior_array[2]
        self.emoexp = behavior_array[3]

        self.rewards_simulated = []
        self.reward = 0

        self.dh_c = 0
        self.tobj_c = 0
        self.decision_r = 0
        self.degree_cost = 0
        self.decision_count = 0


    #======================== FUNCTION of reward equation ========================
    def reward_equation(self):
        #reward equation
        reward_simulated = 0
        simulated_reward = []
        #=========== Setting Parameters ===============
        phi = 2
        delta = 0.09
        alpha = 1.58734
        gamma = 1.27006
        beta = 0.5112

        reward_simulated = - alpha*self.dh_c + gamma*((self.tobj_c/self.tdecide_c)**(1/phi)) + beta* self.decision_r - delta*self.degree_cost
        self.reward = reward_simulated

        #parts of reward equation
        socialsignal_simpart1 = - alpha*self.dh_c
        socialsignal_simpart2 = + gamma*((self.tobj_c/self.tdecide_c)**(1/phi))
        decisioncost = + beta*self.decision_r
        cost_sim = - delta*self.degree_cost

        simulated_reward = [reward_simulated,socialsignal_simpart1,socialsignal_simpart2,decisioncost, cost_sim]
        self.rewards_simulated = simulated_reward


    #============================ DEGREE FUNCTION ========================
    def getdegreeinfo(self):
        degree = 0
        for i in range(len(self.behavior_array)):
            if self.behavior_array[i] == 1:
                degree += 1
        return degree
    #======================== Function of reward simulation================
    def getsimulations(self):
        dh=[]
        tobj=[]
        tdecide=[]
        simulations_array = []

        self.degree_cost = self.getdegreeinfo()

        #=========== Setting Parameters ===============
        #Distances held by users
        mu_dprox, sigma_dprox = 0.801, 0.112
        # mean and standard deviation WITH Proxemics
        mu_dno, sigma_dno = 1.03, 0.216
        # mean and standard deviation WITHOUT Proxemics
        #Time Looking at Object
        mu_tobjyes, sigma_tobjyes = 4452.778, 1714
        # mean and standard deviation TOOK recommendation
        mu_tobjno, sigma_tobjno = 2164.24, 1032.25
        # meand and stdv NOT taken recommendation
        #Time to Make Decision
        mu_tdecide, sigma_tdecide = 3111, 1817
        # mean and standard deviation STANDARD decision time
        mu_tdecidelonger, sigma_tdecidelonger = 4314.619, 2026
        # mean and standard deviation LONGER decision time
        decisions_options = np.array([1, 2])


        #------ SIMULATED DISTANCES HELD BY USER ------------
        dprox = np.random.normal(mu_dprox, sigma_dprox, 80)
        dno = np.random.normal(mu_dno, sigma_dno, 20)
        dprox = list(filter(lambda x : x > 0.52, dprox))
        dno = list(filter(lambda x : x > 0.715, dno))
        #role of proxemics
        if self.prox == 1:
            dh = dprox
        else:
            dh = dno

        self.dh_c = np.random.choice(dh)

        #-------- SIMULATED DECISIONS OF USER -------------
        decisions_altered = []
        decision_seq =[]
        notchoicerobot=0
        choicerobot = 0
        decision = 0

        choicerobot_index = random.randint(0, len(decisions_options)-1)

        for i in range(len(decisions_options)):
            if i == choicerobot_index:
                choicerobot = decisions_options[choicerobot_index]
            else:
                notchoicerobot = decisions_options[i]

        decisions_altered =[choicerobot, notchoicerobot]

        #role of proxemics and emotional expression: more like to take decision similar to robot
        if self.prox==1 or self.emoexp==1:
            #actions that include proxemics or emotional expressions
            decision_seq= random.choices(decisions_altered,weights=(65,35),k=1)
        else:
            decision_seq= random.choices(decisions_altered,weights=(20,80),k=1)
            #less weight on the choice of the robot

        decision = decision_seq[0]

        if decision == choicerobot :
            self.decision_r = 1
            self.decision_count = 1
        else:
            self.decision_r = -1
            self.decision_count =0 

        #-------- SIMULATED TIME LOOKING AT OBJECT BASED ON SIMULATED DECISION -------------
        tobjyes = np.random.normal(mu_tobjyes, sigma_tobjyes, 100)
        tobjno = np.random.normal(mu_tobjno, sigma_tobjno, 100)

        if self.decision_r == 1:
            tobj = tobjyes
        else:
            tobj = tobjno
        # Remove Negative Elements in List --- Using filter() + lambda
        tobj = list(filter(lambda x : x > 0, tobj))
        self.tobj_c = np.random.choice(tobj)

        #----------------- SIMULATED DECISION TIME OF USER -----------------------------------
        tdecide_shorter = np.random.normal(mu_tdecide, sigma_tdecide, 100)
        tdecide_longer = np.random.normal(mu_tdecidelonger, sigma_tdecidelonger, 100)

        if  self.gest== 1 or self.gaze==1:
            #in presence of gesture and/or gaze, decision time is shorter
            tdecide = tdecide_shorter
        else:
            #in absence of gesture and/or gaze, decision time is longer
            #(decreasing the reward from the fraction time_looking/time_deciding)
            tdecide = tdecide_longer

        tdecide = list(filter(lambda x : x > 0, tdecide))
        self.tdecide_c = np.random.choice(tdecide)


    def getrewards(self):
        #------------ CALCULATED REWARD BASED ON SIMULATED HUMAN ------------------------
        self.getsimulations()
        self.reward_equation()
