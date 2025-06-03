### DDPG APPLIED TO modSTREAmS

#Add to python path
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir)
)
sys.path.append(PROJECT_ROOT)

#Imports for DDPG
import numpy as np
import matplotlib.pyplot as plt
import torch
#Imports for RainClouds plots
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid",font_scale=2)
import ptitprince as pt
import torch.nn as nn


def DDPGcontrol(env, Episodes, random_steps, max_episode_steps, update_freq, Learnings):
    from Control_Method import DDPG

    state_dim = env.observation_space.shape[0] #Dimension of the state space (in this case three continous values for x, y, and z of the Lorenz System)
    action_dim = env.action_space.shape[0] #Dimension of the action space (in this case continuous values between 0 and 50)
    max_action = float(env.action_space.high[0]) #The max forcing that can be applied

    agent = DDPG.ddpg(state_dim, action_dim, max_action) #Initializes the DDPG algorithm (Class containing the Actor-Critic NN that chooses x forcing action)
    replay_buffer = DDPG.ReplayBuffer(state_dim, action_dim) #Initializes the ReplayBuffer (stores s,a,s_,r sequences for batch learning)

    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration


    for total_learnings in range(Learnings):
        print("This is learning ", str(total_learnings + 1))

        #Code below resets network parameters to original set between Learnings, making learnings independent of each other but helping to eliminate random inter-learning variation
        if total_learnings > 0:
            # #Place before and after desired network below to print weights before and after reinitialization            
            # print("Learning's Actor params before init")
            # for mod1 in agent.actor.modules():
            #     if isinstance(mod1, nn.Linear):
            #         print(mod1.weight)
            # print("Learning's Actor params after init")
            # for mod2 in agent.actor.modules():
            #     if isinstance(mod2, nn.Linear):
            #         print(mod2.weight)

            OriginalActorParams = torch.load(f'{agent.run_name}/Initial_Parameters/InitialActorParameters.pt')
            agent.actor.load_state_dict(OriginalActorParams)
            OriginalActorTargetParams = torch.load(f'{agent.run_name}/Initial_Parameters/InitialActorTargetParameters.pt')
            agent.actor_target.load_state_dict(OriginalActorTargetParams)
            OriginalCriticParams = torch.load(f'{agent.run_name}/Initial_Parameters/InitialCriticParameters.pt')
            agent.critic.load_state_dict(OriginalCriticParams)    
            OriginalCriticTargetParams = torch.load(f'{agent.run_name}/Initial_Parameters/InitialCriticTargetParameters.pt')
            agent.critic_target.load_state_dict(OriginalCriticTargetParams)      

        for Episode in range(Episodes):
            #Generate initial value for episode (note: forced ICs same as Unforced ICs)
            s = env.reset()
            state_data = torch.tensor(s)
 
            s_noforcing = torch.clone(s)
            state_data_noforcing = s_noforcing.view(1,3)

            #Decriments the random_steps value so fewer are taken each episode (Consider if random steps are actually needed)
            if random_steps >= 0:
                random_steps += -25

            for episode_steps in range(max_episode_steps):

                #Randomly select, or have the NN choose, an action for the current step (i.e. forcing value on x, y, or z)
                if episode_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                    a = env.action_space.sample()
                    a = torch.from_numpy(a)
                else:
                    # Add Gaussian noise to actions for exploration
                    a = agent.choose_action(s)
                    a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
                    a = torch.from_numpy(a)

                s_, r, terminated, truncated = env.step(a, s)
                state_data = torch.cat([state_data.view(-1,3), s_.view(1,3)])

                #Calculate the average MSE for each episode with forcing (EpAveMSE) and w/o forcing (EpAveMSE_NF)
                if episode_steps == 0:
                    print('Start of Episode ' + str(Episode+1))
                    EpAveMSE = (((abs(s[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                    EpAveMSE_NF = (((abs(s_noforcing[0] - env.Ftarget[0])+abs(s[1] - env.Ftarget[1])+abs(s[2] - env.Ftarget[2]))**2))
                else:
                    EpAveMSE = ((EpAveMSE*(episode_steps-1)) + (abs(s[0] - env.Ftarget[0])**2+abs(s[1] - env.Ftarget[1])**2+abs(s[2] - env.Ftarget[2])**2))/episode_steps
                    EpAveMSE_NF = ((EpAveMSE_NF*(episode_steps-1)) + (abs(s_noforcing[0] - env.Ftarget[0])**2+abs(s_noforcing[1] - env.Ftarget[1])**2+abs(s_noforcing[2] - env.Ftarget[2])**2))/episode_steps

                #Executes after the final step in each episode    
                if episode_steps == max_episode_steps-1:
                    LearningAveMSE.append(EpAveMSE)
                    LearningAveMSE_NF.append(EpAveMSE_NF)
                    if EpAveMSE == min(LearningAveMSE):

                        #Store the State Data for the Forcing and Non-Forcing Cases as np arrays for plotting
                        BSD = np.array(state_data)
                        UFSD = np.array(state_data_noforcing)

                        #Create path to parameter file
                        Control_Imp = os.path.dirname(os.path.abspath(__file__))
                        FluidML = os.path.abspath(os.path.join(Control_Imp, '..'))
                        best_params_dir = os.path.join(FluidML, agent.run_name)
                        best_params_dir = os.path.join(best_params_dir, 'Best_Parameters', f'BestParams_Learning{total_learnings+1}')

                        if not os.path.exists(best_params_dir):
                            os.makedirs(best_params_dir)

                        for file_name in os.listdir(best_params_dir):
                            file_path = os.path.join(best_params_dir, file_name)
                            if os.path.isfile(file_path):
                                os.remove(file_path)

                        myfilepath1 = os.path.join(best_params_dir, f'BestParamsActor_Learning{total_learnings+1}.pt')
                        myfilepath2 = os.path.join(best_params_dir, f'BestParamsCritic_Learning{total_learnings+1}.pt')
                        
                        torch.save(agent.actor.state_dict(), myfilepath1)
                        torch.save(agent.critic.state_dict(), myfilepath2)

                    #Print the average Mean Squared Error for the Episode
                    print('Episode Ave MSE:', EpAveMSE)
                    print()

                #if terminated:
                    #break
                replay_buffer.store(s, a, r, s_)  # Store the transition
                s = s_

                # Take 50 steps,then update the networks 50 times
                if episode_steps >= random_steps and episode_steps % update_freq == 0:
                    for _ in range(update_freq):
                        agent.learn(replay_buffer)

        #Store the state data for the best performing policy of each Learning 
        Best_State_Data_Storage = {}
        Best_State_Data_Storage['Learning_' + str(total_learnings + 1)] = BSD
        Unforced_State_Data_Storage = {}
        Unforced_State_Data_Storage['Learning_' + str(total_learnings + 1)] = UFSD
        
        plot_storage = os.path.join(FluidML, agent.run_name)
        plot_storage = os.path.join(plot_storage, 'Training_Plots', f'Learning{total_learnings+1}_Plots')
        os.makedirs(plot_storage)


