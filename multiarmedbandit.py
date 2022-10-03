#import modules
import numpy as np
import matplotlib.pyplot as plt
import random
np.set_printoptions(precision=5)
#from rewardfunction import Reward_Function
import pandas as pd
#%matplotlib inline

'''
reward function:
R = = - alpha*dh + gamma*(sqrt(tobj/tdecide)) + beta* decision - delta*degree
'''
#=========== Setting Parameters ===============
choice_robot = 1

#alpha = 16.160365
#gamma = 14.59832
phi = 2
#delta =0.24
delta = 0.09
#beta = 1.8834
alpha = 1.58734
gamma = 1.27006
beta = 0.50112

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

#======================== FUNCTION of reward equation ========================
def reward_equation(dh_c, tobj_c,tdecide_c, decision, degree_cost):
    #reward equation
    #reward_simulated = - alpha*dh_c + gamma*((tobj_c/tdecide_c)**(1/phi))
    reward_simulated = - alpha*dh_c + gamma*((tobj_c/tdecide_c)**(1/phi)) + beta* decision - delta*degree_cost

    #parts of reward equation
    #socialsignal_simpart1 = - alpha*dh_c
    #socialsignal_simpart2 = + gamma*((tobj_c/tdecide_c)**(1/phi))
    #decisioncost = + beta*decision
    #cost_sim = - delta*degree_cost

    socialsignal_simpart1 = dh_c
    socialsignal_simpart2 = tobj_c
    decisioncost = tdecide_c
    if decision ==1 :
        cost_sim = 1
    else:
        cost_sim = 0

    simulated_reward = [reward_simulated,socialsignal_simpart1,socialsignal_simpart2,decisioncost, cost_sim]

    return simulated_reward

#======================== INFORMATION OF ACTION ========================
def getactioninfo(action):
    info_action = []
    degree = 0
    proxemics_presence = 0
    proxemotionalexp_presence = 0
    gazegest_presence = 0

    if action >=0 and action<4:
        degree = 1
    elif action >=4 and action<10:
        degree = 2
    elif action >= 10 and action<14:
        degree = 3
    else:
        degree = 4

    #presence of proxemics in actions
    if action == 0 or action == 4 or action ==5 or action ==6 or action ==10 or action==11 or action==12 or action== 14:
        proxemics_presence = 1
    else:
        proxemics_presence = 2
    # presence of proxemics and/or emotional expressions in actions
    if not(action == 1 or action== 2 or action==7):
        proxemotionalexp_presence = 1
    else:
        proxemotionalexp_presence = 2

    #presence of gestures and/or gaze in actions
    if not (action==0 or action==3 or action==6):
        gazegest_presence = 1
    else:
        gazegest_presence = 2

    info_action = [degree, proxemics_presence, proxemotionalexp_presence, gazegest_presence]

    return info_action
#======================== Function of reward simulation================
def reward_simulations(action):
    simulated_reward = 0
    action_info = []
    dh=[]
    tobj=[]
    tdecide=[]
    decision_r = 0
    degree_cost = 0
    proxemics_presence = 0
    proxemotionalexp_presence = 0
    gazegest_presence = 0

    action_info = getactioninfo(action)

    degree_cost = action_info[0]
    proxemics_presence = action_info[1]
    proxemotionalexp_presence = action_info[2]
    gazegest_presence = action_info[3]

    #------ SIMULATED DISTANCES HELD BY USER ------------
    dprox = np.random.normal(mu_dprox, sigma_dprox, 80)
    dno = np.random.normal(mu_dno, sigma_dno, 20)
    dprox = list(filter(lambda x : x > 0.52, dprox))
    dno = list(filter(lambda x : x > 0.715, dno))
    #role of proxemics
    if proxemics_presence == 1:
        dh = dprox
    else:
        dh = dno

    dh_c = np.random.choice(dh)

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
    if proxemotionalexp_presence==1:
        #actions that include proxemics or emotional expressions
        decision_seq= random.choices(decisions_altered,weights=(65,35),k=1)
    else:
        decision_seq= random.choices(decisions_altered,weights=(35,65),k=1)
        #less weight on the choice of the robot

    decision = decision_seq[0]

    if decision == choice_robot :
        decision_r = 1
    else:
        decision_r = -1

    #-------- SIMULATED TIME LOOKING AT OBJECT BASED ON SIMULATED DECISION -------------
    tobjyes = np.random.normal(mu_tobjyes, sigma_tobjyes, 100)
    tobjno = np.random.normal(mu_tobjno, sigma_tobjno, 100)

    if decision_r == 1:
        tobj = tobjyes
    else:
        tobj = tobjno
    # Remove Negative Elements in List --- Using filter() + lambda
    tobj = list(filter(lambda x : x > 0, tobj))
    tobj_c = np.random.choice(tobj)

    #----------------- SIMULATED DECISION TIME OF USER -----------------------------------
    tdecide_shorter = np.random.normal(mu_tdecide, sigma_tdecide, 100)
    tdecide_longer = np.random.normal(mu_tdecidelonger, sigma_tdecidelonger, 100)

    if gazegest_presence == 1:
        #in presence of gesture and/or gaze, decision time is shorter
        tdecide = tdecide_shorter
    else:
        #in absence of gesture and/or gaze, decision time is longer
        #(decreasing the reward from the fraction time_looking/time_deciding)
        tdecide = tdecide_longer

    tdecide = list(filter(lambda x : x > 0, tdecide))
    tdecide_c = np.random.choice(tdecide)

    #------------ CALCULATED REWARD BASED ON SIMULATED HUMAN ------------------------
    simulated_reward = reward_equation(dh_c, tobj_c,tdecide_c, decision_r, degree_cost)

    return simulated_reward

class Bandit:
    '''
    input:
    k: number of bandits we hope to inialise
    exp_rate: exploration rate
    lr: learning rate
    UCB method with value c (Upper - Confidence-Bound)

    to define:
    actions: represented as numbers of combinations of multi-modal behaviors the robot can perform
    initial estimates of values are set to 0
    '''

    def __init__(self, k=15, exp_rate=.3, lr=0.1, ucb=False, seed=None, c=2):
        self.k = k
        self.actions = range(self.k)
        self.exp_rate = exp_rate
        self.lr = lr
        self.total_reward = 0
        self.avg_reward = []
        self.standard_dev = []
        #self.standard_devsocialsig = []
        #self.total_costdegree = 0
        #self.avg_costdegree =[]
        #self.total_socialsig = 0
        #self.avg_socialsig = []

        self.total_decision = 0
        self.avg_decision = []

        self.total_distance = 0
        self.avg_distance = []
        self.stdv_distance = []

        self.total_timelooking = 0
        self.avg_timelooking = []
        self.stdv_timelooking = []

        self.total_timedeciding = 0
        self.avg_timedeciding = []
        self.stdv_timedeciding = []

        self.total_countdecisions =0
        self.avg_countdecisions = []
        self.stdv_countdecisions = []

        self.TrueValue = []
        np.random.seed(seed)

        self.rewardperaction = [0]*self.k

        #implementing UCB
        self.values = np.zeros(self.k)
        #total number of trials
        self.times = 0
        #counts number of actions in each bandit
        self.action_times = np.zeros(self.k)

        #if selection action using upper-confidence bound
        self.ucb = ucb
        self.c = c


    def chooseAction(self):
        # exploration based on the exploration rate
        # c= 2 here and self.action_times is added by 0.1 in case the value is divided by 0
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # exploit
            if self.ucb:
                if self.times == 0:
                    action = np.random.choice(self.actions)
                else:
                    confidence_bound = self.values + self.c * np.sqrt(
                        np.log(self.times) / (self.action_times + 0.1))  # c=2
                    action = np.argmax(confidence_bound)
            else:
                action = np.argmax(self.values)
        return action


    def takeAction(self,action):
        self.times +=1
        self.action_times[action] += 1

        self.TrueValue = reward_simulations(action)
        #reward = self.TrueValue[0] + np.random.normal(self.TrueValue[0], 0.08)
        #socialsig = self.TrueValue[1] + self.TrueValue[2]
        #decision = self.TrueValue[3]
        #costdegree = self.TrueValue[4]

        reward = self.TrueValue[0]
        distances_sim = self.TrueValue[1]
        timelooking_sim = self.TrueValue[2]
        timedeciding_sim = self.TrueValue[3]
        countdecisions_sim = self.TrueValue[4]

        self.rewardperaction[action] = (reward + self.rewardperaction[action])/(self.action_times[action])

        # using incremental method to propagate
        self.values[action] += self.lr * (reward - self.values[action])  # look like fixed lr converges better

        self.total_reward += reward
        self.avg_reward.append(self.total_reward / self.times)
        self.standard_dev.append(np.std(self.avg_reward))

        self.total_distance += distances_sim
        self.avg_distance.append(self.total_distance / self.times)
        self.stdv_distance.append(np.std(self.avg_distance))

        self.total_timelooking += timelooking_sim
        self.avg_timelooking.append(self.total_timelooking / self.times)
        self.stdv_timelooking.append(np.std(self.avg_timelooking))

        self.total_timedeciding += timedeciding_sim
        self.avg_timedeciding.append(self.total_timedeciding / self.times)
        self.stdv_timedeciding.append(np.std(self.avg_timedeciding))

        self.total_countdecisions += countdecisions_sim
        self.avg_countdecisions.append(self.total_countdecisions / self.times)
        self.stdv_countdecisions.append(np.std(self.avg_countdecisions))

        #self.total_socialsig += socialsig
        #self.avg_socialsig.append(self.total_socialsig/self.times)
        #self.standard_devsocialsig.append(np.std(self.avg_socialsig))
        #self.total_costdegree += costdegree
        #self.avg_costdegree.append(self.total_costdegree/self.times)
        #self.total_decision += decision
        #self.avg_decision.append(self.total_decision/self.times)

    def play(self, n):
        for _ in range(n):
            action = self.chooseAction()
            self.takeAction(action)
