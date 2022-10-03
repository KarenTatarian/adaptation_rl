# creating a RL environment
# going to use OpenCV installed using pip install opencv-python
import numpy as np  # for array stuff and random
import pandas as pd
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.
from rewardfunction import Reward_Function
from multiarmedbandit import Bandit
import random
np.set_printoptions(precision=5)
import seaborn as sns
import time

style.use("ggplot")  # setting our style!

SIZE = 1+1 #grid environment instead of discrete observational environment
HM_EPISODES = 15_000 #how many EPISODES
NUM_RUNS = 10

##****** EPSILON **********************
# First: epsilon at 0.9 and register q-table
# Then once we have q-table, change start_q_table = None to "nameoffile"
#AND change epsilon to 0
epsilon = 0.99
#EPS_DECAY = 1
EPS_DECAY = 0.9998 # decided this number randomly # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 100 #How often to show

start_q_table = None
#start_q_table = "after25k.pickle" # or filename # FIRST NONE
LEARNING_RATE = 0.2
DISCOUNT = 0.95
# can ticker with these number using the analysis tools
eps01lr02_rewards = np.zeros(HM_EPISODES)
eps01lr02_stdv  = np.zeros(HM_EPISODES)

class Robot:
    def __init__(self): #where to start
    #start with random combination of modalities
        self.prox = np.random.randint(0,SIZE) #position of the blolb
        self.gest = np.random.randint(0,SIZE)
        self.gaze = np.random.randint(0,SIZE)
        self.emoexp = np.random.randint(0,SIZE)

    def __str__(self): # for debugging purposes
        return f"{self.prox}, {self.gest},{self.gaze},{self.emoexp}" # print combination

    def obs(self):
        return (self.prox, self.gest, self.gaze, self.emoexp)

    def action(self, choice): #8 possible discrete actions
        if choice == 0:
            self.combine(x=1,y=0, z=0,w=0)
        elif choice == 1:
            self.combine(x=-1,y=0, z=0,w=0)
        elif choice == 2:
            self.combine(x=0,y=1, z=0,w=0)
        elif choice == 3:
            self.combine(x=0,y=-1, z=0,w=0)
        elif choice == 4:
            self.combine(x=0,y=0, z=1,w=0)
        elif choice == 5:
            self.combine(x=0,y=0, z=-1,w=0)
        elif choice == 6:
            self.combine(x=0,y=0, z=0,w=1)
        if choice == 7:
            self.combine(x=0,y=0, z=0,w=-1)

    def combine(self, x=False, y=False, z=False, w=False):
        # x for changing Proxemics
        if x!=0:
            self.prox += x
        # y for Gesture
        if y!=0:
            self.gest += y
        #z for gaze
        if z!=0:
            self.gaze += z
        #w for Emotional Expressions
        if w!=0:
            self.emoexp += w

        # stop it from going beyond the {0,1} for the modalities
        if self.prox <0:
            self.prox = 0
        elif self.prox > SIZE-1:
            self.prox = SIZE -1
        if self.gest <0:
            self.gest = 0
        elif self.gest > SIZE-1:
            self.gest = SIZE -1
        if self.gaze <0:
            self.gaze = 0
        elif self.gaze > SIZE-1:
            self.gaze = SIZE -1
        if self.emoexp <0:
            self.emoexp = 0
        elif self.emoexp > SIZE-1:
            self.emoexp = SIZE -1

#------------------ CREATE Q-TABLE -----------------------
# The observation space is : (x1, y1),(x2,y2) [four tuples]
#so to get every combination we need to iterate through all of them
def create_q_table():
    #if start_q_table is None:
    q_table = {} #making it a dictionary
    for prox_i in range(0, SIZE): #(-size +1) to shift
        for gest_i in range(0, SIZE):
            for gaze_i in range(0, SIZE):
                for emoexp_i in range(0, SIZE):
                    # action space is 8 discrete actions, so need 8 random variables (iniatiting with random values)
                    q_table[(prox_i,gest_i,gaze_i,emoexp_i)] = [np.random.uniform(-0.1,0) for i in range(8)]
    #else:
        #with open(start_q_table, "rb") as f: # if we have one already pretrained
            #q_table = pickle.load(f)
    return q_table

def average_qtable():
    q_tableaverage = {} #making it a dictionary
    for prox_i in range(0, SIZE): #(-size +1) to shift
        for gest_i in range(0, SIZE):
            for gaze_i in range(0, SIZE):
                for emoexp_i in range(0, SIZE):
                    # action space is 8 discrete actions, so need 8 random variables (iniatiting with random values)
                    q_tableaverage[(prox_i,gest_i,gaze_i,emoexp_i)] = [np.random.uniform(-0.1,0) for i in range(8)]
    return q_tableaverage
##----------------------------------- CREATING DICTIONARY TO STORE INFORMATION FOR REWARD-------------------
def create_dictionaryrewards():
    dictionary_rewards = {}
    # DICTIONARY OF REWARDS : {times state was visited, total reward from that state so far, array of moving avg reward of that state}
    for prox_i in range(0, SIZE): #(-size +1) to shift
        for gest_i in range(0, SIZE):
            for gaze_i in range(0, SIZE):
                for emoexp_i in range(0, SIZE):
                    dictionary_rewards[(prox_i,gest_i,gaze_i,emoexp_i)] =[0,0,[]]
    return dictionary_rewards

# ----------------------------- TRAINING FOR Q-TABLE ---------------------------
dictionary_episoderewards = []
dictionary_timesvisited = []
print_states =[]
q_tableaverage = average_qtable()
seed = 1234
starttime_Qlearning = time.time()
timepisode_qlearning = []
print(starttime_Qlearning)
for i in range(NUM_RUNS):
    np.random.seed(seed)
    episode_rewards = []
    print_timesvisited = []
    q_table = create_q_table()
    dictionary_rewards = create_dictionaryrewards()
    epsilon = 0.99
    starttime_eps_qlearning = time.time()
    for episode in range(HM_EPISODES):
        rob_behavior = Robot()
        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon is {epsilon}") #to know where we are
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        episode_reward = 0
        moving_avgepisode = 0
        for i in range(50): # the 200 can be hard coded
            #define observation
            obs = rob_behavior.obs()
            #random movement (action selection)
            if np.random.random() > epsilon:
                #regular action
                action = np.argmax(q_table[obs])
            else:
                #exploration
                action = np.random.randint(0,8) #0,1,2,3
            #applying action
            rob_behavior.action(action)
            ############

            # -------------- REWARD -------------
            new_obs = rob_behavior.obs()
            reward_funct = Reward_Function(behavior_array=new_obs)
            reward_funct.getrewards()
            reward_newobs = reward_funct.reward

            #--- update dictionary of that state -----
            dictionary_rewards[new_obs][0] +=1 #times state visited
            dictionary_rewards[new_obs][1] += reward_newobs #total reward from state visited
            moving_avgepisode = (dictionary_rewards[new_obs][1])/(dictionary_rewards[new_obs][0])
            dictionary_rewards[new_obs][2].append(moving_avgepisode)

            # UPDATING Q-TABLE
            max_future_q = np.max(q_table[new_obs])
            current_q = q_table[obs][action]

            new_q = (1- LEARNING_RATE)*current_q + LEARNING_RATE*(reward_newobs + DISCOUNT*max_future_q)

            q_table[obs][action] = new_q
            episode_reward += reward_newobs

        episode_rewards.append(episode_reward)
        # DECAY EPSILON
        epsilon *= EPS_DECAY

    timepisode_qlearning.append(time.time() - starttime_eps_qlearning )
    print(f"on #{episode}, epsilon is {epsilon}")

    #Moving average
    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode="valid")
    plotting_movingavg = []
    for i in range(len(moving_avg)):
        plotting_movingavg.append(moving_avg[i])
    #dictionary_episoderewards[i] = plotting_movingavg
    dictionary_episoderewards.append(moving_avg)

    for k, v in dictionary_rewards.items():
        timev, reward, array = v
        print_states.append(k)
        print_timesvisited.append(timev)
    dictionary_timesvisited.append(print_timesvisited)

    for prox_i in range(0, SIZE): #(-size +1) to shift
        for gest_i in range(0, SIZE):
            for gaze_i in range(0, SIZE):
                for emoexp_i in range(0, SIZE):
                    for action in range(8):
                        q_tableaverage[(prox_i,gest_i,gaze_i,emoexp_i)][action] += q_table[(prox_i,gest_i,gaze_i,emoexp_i)][action ]
    with open(f"q-tablerun{i}.pickle", "wb") as f:
        pickle.dump(q_table, f)


timingqlearning = time.time() - starttime_Qlearning

starttime_mab = time.time()
time_mab = []
####---------- MULTI-ARMED BANDIT -----------
for i in range(NUM_RUNS):
    start_eps_mab = time.time()
    bdt_eps01lr02 = Bandit(k=15, exp_rate=0.1, lr=0.2, seed=1234)
    bdt_eps01lr02.play(HM_EPISODES)
    #### Get action selection and average reward for bandit run
    #---------------------------------------------------------
    avg_reward_eps01lr02 = bdt_eps01lr02.avg_reward
    stdv_eps01lr02 = bdt_eps01lr02.standard_dev

    ##### Update long-term average rewards
    #---------------------------------------------------------
    eps01lr02_rewards = eps01lr02_rewards + (avg_reward_eps01lr02 - eps01lr02_rewards) / (i + 1)
    eps01lr02_stdv = eps01lr02_stdv + (stdv_eps01lr02 - eps01lr02_stdv) / (i + 1)
    time_mab.append(time.time() - start_eps_mab)
#endtime_mab = time.time()
timingmab = time.time() - starttime_mab

print(f"TIME FOR Q-LEARNING {timingqlearning}")
print(f"QLEARNING: AVERAGE TIME FOR ONE EPISODE WITH 50 STEPS each {timingqlearning/NUM_RUNS}")
print(f"MAB: TIME FOR EPSILON-MAB {timingmab}")
print(f"MAB: average TIME FOR EPSILON-MAB FOR ONE RUN {timingmab/NUM_RUNS}")

averagetime_eps_qlearning = np.mean(timepisode_qlearning)
stdvtime_eps_qlearning = np.std(timepisode_qlearning)
averagetime_eps_mab = np.mean(time_mab)
stdvtime_eps_mab = np.std(time_mab)
print(f"QLEARNING: AVERAGE TIME FOR ONE RANDOM SEED RUN OF {HM_EPISODES} IS {averagetime_eps_qlearning} with a stdv {stdvtime_eps_qlearning}")
print(f"MAB: AVERAGE TIME FOR ONE RANDOM SEED RUN OF {HM_EPISODES} IS {averagetime_eps_mab} with a stdv {stdvtime_eps_mab}")

### --------- PLOTTING THE AVERAGE REWARD FUNCTION AND THE VARIANCE BETWEEN RUNS(EPOCHS?)-------
avg = 0
variance = 0
avg_plotting = []
varriance_plotting = []
for y in range(len(dictionary_episoderewards[0])):
    array = []
    for i in range(NUM_RUNS):
        array.append(dictionary_episoderewards[i][y])
    avg = np.mean(array)
    variance = np.std(array)
    #variance = np.var(array)
    avg_plotting.append(avg)
    varriance_plotting.append(variance)

# Plot
kwargs = dict(kde_kws={'linewidth':2})
sns.set_style("whitegrid")

def tsplot(data,style='-',labelofdata='',**kw):
    x = np.arange(data.shape[0])
    est = data.avg_reward
    sd = data.varriance
    cis = (est - sd, est + sd)
    plt.fill_between(x,cis[0],cis[1],alpha=0.1, **kw)
    plt.plot(x,est, marker=style,markevery=500, **kw, label=labelofdata,linewidth=2.5)
    plt.margins(x=0)

def create_df(eps_rewards = [], eps_varr =[]):
    x_01 = np.array(range(1, len(eps_rewards)+1))
    data = {'time': x_01,
           'avg_reward': eps_rewards,
           'varriance': eps_varr}
    df = pd.DataFrame(data)
    return df

df_qlearning = create_df(avg_plotting, varriance_plotting)

#PLOTTING REWARD FUNCTION
plt.figure(figsize=(12,8), dpi= 200)
plt.rcParams.update({'font.size': 20})
tsplot(df_qlearning,'*',r'$Q-Learning(\epsilon-decay)$')
plt.legend(loc='upper right', ncol=3, handlelength=1, borderaxespad=0.,prop={'size':14})
#plt.title("Average Reward per Algorithm at 15000 Iterations",size=18)
plt.xlabel("Episodes", size=18)
plt.ylabel("Average Episode Reward", size=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.savefig("15thou10randomseedruns_avg.png",dpi=200)
plt.show()


## ------------------ PLOTTING THE AVERAGE TIMES EACH STATE IS VISIED -------
avg_times = 0
std_times = 0
avg_plottingtimes = []
std_plottingtimes =[]
TOTAL_STATES = 16
for state in range(TOTAL_STATES):
    array = []
    for i in range(NUM_RUNS):
        array.append(dictionary_timesvisited[i][state])
    avg_times = np.mean(array)
    std_times = np.std(array)
    avg_plottingtimes.append(avg_times)
    std_plottingtimes.append(std_times)

percentage_timesvisited=[]
total_timesvisited=0
percentage_stdvtimesvisited =[]
for i in range(len(avg_plottingtimes)):
    total_timesvisited=total_timesvisited + avg_plottingtimes[i]
for i in range(len(avg_plottingtimes)):
    percentage_timesvisited.append((avg_plottingtimes[i]*100)/total_timesvisited)
    percentage_stdvtimesvisited.append((std_plottingtimes[i]*100)/total_timesvisited)

#length = len(print_states)
#middle_index = length//2
if len(print_states) >16:
    states_array = print_states[:16]
else:
    states_array = print_states


d = {'Percentage of Times Visited': percentage_timesvisited, 'States':[" " + str(k) for k in states_array ], 'sd': percentage_stdvtimesvisited}
df_times = pd.DataFrame(data=d)
df_times

plt.figure(figsize=(18,10))
plt.rcParams.update({'font.size': 25})
sns.barplot(x="Percentage of Times Visited",y="States", data =df_times,ci= "sd", palette="Blues_d")
plt.title("Percentage of Times Each State was Visited")
plt.savefig("15thou10random_percentagestates.png")
plt.show()

##---------- PLOTTING MULTI-ARMED BANDIT AND Q-LEARNING ------
moving_avgMAB = np.convolve(eps01lr02_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode="valid")
moving_stdvMAB= np.convolve(eps01lr02_stdv, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode="valid")
plotting_movingavgMAB = []
plotting_stdvMAB =[]
for i in range(len(moving_avgMAB)):
    plotting_movingavgMAB.append(moving_avgMAB[i])
    plotting_stdvMAB.append(moving_stdvMAB[i])

df_mab = create_df(plotting_movingavgMAB, plotting_stdvMAB)
plt.figure(figsize=(12,8), dpi= 200)
plt.rcParams.update({'font.size': 20})
tsplot(df_mab,'o',r'$\epsilon-MAB(\epsilon=0.1,\alpha=0.2)$')
tsplot(df_qlearning,'*',r'$Q-Learning(\epsilon-decay)$')
plt.legend(loc='upper right', ncol=3, handlelength=1, borderaxespad=0.,prop={'size':14})
#plt.title("Average Reward per Algorithm at 15000 Iterations",size=18)
plt.xlabel("Iterations", size=18)
plt.ylabel("Average Episode Reward", size=18)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.savefig("15thou10random_bothplots.png",dpi=200)
plt.show()


###### ----- PLOTTING THE HEAT MAP OF Q(S,A) ----------
# MANUALLY CODED
total_states = 16
total_actions = 8
p = []
for i in range(total_states):
    p.append(np.zeros(total_actions))
pp_average = np.array(p)
pp = np.array(p)

state = -1
for k, v in dictionary_rewards.items():
    time,reward,array = v
    state = state +1
    for a in range(total_actions):
        pp[state][a] = q_table[k][a]
        pp_average[state][a] = (q_tableaverage[k][a]/NUM_RUNS)
#print(pp)
print_actions =["proxemics=+1", "proxemics=-1", "gesture=+1", "gesture=-1","gaze=+1", "gaze=-1", "emotionalexp=+1","emotionalexp=-1"]
#print(len(pp))
df = pd.DataFrame(pp, index=[" " + str(k) for k in states_array],
                 columns=[" " + str(x) for x in print_actions])

plt.figure(figsize=(35,45))
plt.rcParams.update({'font.size': 40})
#print(total_states)
res = sns.heatmap(data=df, annot=True, linewidth=0.4, cmap="mako", annot_kws={"size":20})
res.set_yticklabels(res.get_ymajorticklabels(), fontsize =40)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize =40)
# Add label for horizontal axis
#plt.title("$Q(s,a)$ after x Episodes")
#plt.xlabel("Actions, multi-modal combinations", size=30)
#plt.ylabel("State of Robot: (proxemoics, gesture, gaze, emotional_expression)", fontsize=30)
plt.savefig("15thou10randomq(s,a).png")
plt.show()

### ---- AVERAGE Q-TABLE ----

df_average = pd.DataFrame(pp_average, index=[" " + str(k) for k in states_array],
                 columns=[" " + str(x) for x in print_actions])

plt.figure(figsize=(35,45))
plt.rcParams.update({'font.size': 40})
#print(total_states)
res = sns.heatmap(data=df_average, annot=True, linewidth=0.4, cmap="mako", annot_kws={"size":20})
res.set_yticklabels(res.get_ymajorticklabels(), fontsize =40)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize =40)
# Add label for horizontal axis
#plt.title("$Q(s,a)$ after x Episodes")
#plt.xlabel("Actions, multi-modal combinations", size=30)
#plt.ylabel("State of Robot: (proxemoics, gesture, gaze, emotional_expression)", fontsize=30)
plt.savefig("15thou10randomAVERAGE_q(s,a).png")
plt.show()
