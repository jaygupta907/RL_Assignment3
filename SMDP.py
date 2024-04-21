import gym
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import time





environment = gym.make("Taxi-v3")
environment.reset()
num_rows = 5
num_col = 5
num_options = 4
num_episodes=5000
epsilon=0.1
alpha = 0.1
gamma = 0.99


def get_action(Q_values,row,column,epsilon):
    p = np.random.rand()
    action = random.randint(0,Q_values.shape[-1]-1) if p<epsilon else np.argmax(Q_values[row,column,:])
    return action


def options(environment,state,Q_values,option_goal,epsilon=0.1):
    goal =  environment.locs
    optdone=False
    row,column,passenger,destination = environment.decode(state)
    if(row==goal[option_goal][0] and column==goal[option_goal][1]):
        optdone = True
        if option_goal==passenger:
            optact = 4
        elif option_goal == destination:
            optact = 5
        else:
            optact = np.random.randint(0,4)
    else:
        optact =  get_action(Q_values[option_goal],row,column,epsilon)
    return [optact,optdone]

def new_options(environment,state,Q_values,option_goal,epsilon=0.1):
    goal=[(2,1)]
    optdone=False
    row,column,_,destination = environment.decode(state)
    if(row==goal[0][0] and column==goal[0][1] ):
        optdone = True
        if destination==0 or destination==2:
            optact = 3
        else:
            optact=2

    else:
        optact =  get_action(Q_values[option_goal],row,column,epsilon)
    return [optact,optdone]






def plot_Q(Q,message):
    DOWN = 1
    UP=0
    RIGHT=2
    LEFT=3
    plt.figure(figsize=(7,7))
    plt.title(message)
    plt.pcolor(Q.max(-1), edgecolors='#ffffff', linewidths=1)
    plt.colorbar()
    def x_direct(a):
        if a in [UP, DOWN]:
            return 0
        return 1 if a == RIGHT else -1
    def y_direct(a):
        if a in [RIGHT, LEFT]:
            return 0
        return -1 if a == UP else 1
    policy = Q.argmax(-1)
    policyx = np.vectorize(x_direct)(policy)
    policyy = np.vectorize(y_direct)(policy)
    idx = np.indices(policy.shape)
    plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
    plt.show()



def SMDP_Q_Learning(environment):
    action_dim = environment.action_space.n
    Q_options = np.zeros((num_options,num_rows,num_col,action_dim-2))
    rewards=[]
    for i in range(num_episodes):
        state=environment.reset()
        done =False
        episode_reward = 0
        while not done:
            row,column,passenger,destination = environment.decode(state)

            find_passenger=False
            option_goal = passenger#get option implemented
            while not find_passenger and not done and option_goal<4:
                optact,find_passenger = options(environment,state,Q_options,option_goal,epsilon)
                row,column,_,_=environment.decode(state)
                next_state,reward,done,_=environment.step(optact)
                row_next,column_next,_,_ = environment.decode(next_state)
                episode_reward+=reward
                if optact<4:
                    Q_options[option_goal,row,column,optact] = Q_options[option_goal,row,column,optact]+\
                        alpha*(reward+gamma*np.max(Q_options[option_goal,row_next,column_next,:])-Q_options[option_goal,row,column,optact])

                state = next_state
                
            row,column,passenger,destination = environment.decode(state)
            find_destination =False
            option_goal  = destination #put option implemented

            while not find_destination and not done:
                optact,find_destination = options(environment,state,Q_options,option_goal,epsilon)
                row,column,_,_=environment.decode(state)
                next_state,reward,done,_=environment.step(optact)
                row_next,column_next,_,_ = environment.decode(next_state)
                episode_reward+=reward          
                if optact<4:
                    Q_options[option_goal,row,column,optact] = Q_options[option_goal,row,column,optact]+\
                        alpha*(reward+gamma*np.max(Q_options[option_goal,row_next,column_next,:])-Q_options[option_goal,row,column,optact])
                state = next_state
            
        rewards.append(episode_reward)
    return rewards,Q_options

rewards,Q_options = SMDP_Q_Learning(environment)
avg_rewards = [np.average(rewards[i:i+100]) for i in range(len(rewards)-100)]
plt.figure(figsize = (10,5))
plt.plot(avg_rewards)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('score averaged over previous 100 runs')
plt.show()


# Options = ['R','G','Y','B']
# for i in range(num_options):
#     plot_Q(np.flip(Q_options[i],axis=0),message="Q Values for option "+Options[i])


new_num_options=5
def SMDP_New_options(environment):

    action_dim = environment.action_space.n
    Q_options = np.zeros((new_num_options,num_rows,num_col,action_dim-2))
    rewards=[]
    for i in range(num_episodes):
        state=environment.reset()
        done =False
        episode_reward = 0
        while not done:
            row,column,passenger,destination = environment.decode(state)

            find_passenger=False
            option_goal = passenger#get option implemented
            while not find_passenger and not done and option_goal<4:
                optact,find_passenger = options(environment,state,Q_options,option_goal,epsilon)
                row,column,_,_=environment.decode(state)
                next_state,reward,done,_=environment.step(optact)
                row_next,column_next,_,_ = environment.decode(next_state)
                episode_reward+=reward
                if optact<4:
                    Q_options[option_goal,row,column,optact] = Q_options[option_goal,row,column,optact]+\
                        alpha*(reward+gamma*np.max(Q_options[option_goal,row_next,column_next,:])-Q_options[option_goal,row,column,optact])

                state = next_state
                
            row,column,passenger,destination = environment.decode(state)
            bottleneck=False
            option_goal=4
            while not bottleneck and not done:
                optact,bottleneck = new_options(environment,state,Q_options,option_goal,epsilon)
                row,column,_,_=environment.decode(state)
                next_state,reward,done,_=environment.step(optact)
                row_next,column_next,_,_ = environment.decode(next_state)
                episode_reward+=reward          
                if optact<4:
                    Q_options[option_goal,row,column,optact] = Q_options[option_goal,row,column,optact]+\
                        alpha*(reward+gamma*np.max(Q_options[option_goal,row_next,column_next,:])-Q_options[option_goal,row,column,optact])
                state = next_state


            find_destination =False
            option_goal  = destination #put option implemented

            while not find_destination and not done:
                optact,find_destination = options(environment,state,Q_options,option_goal,epsilon)
                row,column,_,_=environment.decode(state)
                next_state,reward,done,_=environment.step(optact)
                row_next,column_next,_,_ = environment.decode(next_state)
                episode_reward+=reward          
                if optact<4:
                    Q_options[option_goal,row,column,optact] = Q_options[option_goal,row,column,optact]+\
                        alpha*(reward+gamma*np.max(Q_options[option_goal,row_next,column_next,:])-Q_options[option_goal,row,column,optact])
                state = next_state
            
        rewards.append(episode_reward)
    return rewards,Q_options


rewards,Q_options = SMDP_New_options(environment)
avg_rewards_new = [np.average(rewards[i:i+100]) for i in range(len(rewards)-100)]
plt.figure(figsize = (10,5))
plt.plot(avg_rewards_new)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('score averaged over previous 100 runs')
plt.show()


Options = ['R','G','Y','B',"Bottleneck"]
for i in range(new_num_options):
    plot_Q(np.flip(Q_options[i],axis=0),message="Q Values for option "+Options[i])



plt.figure(figsize = (10,5))
plt.plot(avg_rewards)
plt.plot(avg_rewards_new)
plt.legend(["With four options(R,G,B,Y)"," with bottleneck option"])
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('score averaged over previous 100 runs')
plt.show()
