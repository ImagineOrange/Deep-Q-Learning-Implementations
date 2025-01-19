#environment imports
import numpy as np
import pygame 
import random
import timeit
start_time = timeit.default_timer()

#memory replay data structures
from collections import deque, namedtuple
import matplotlib.pyplot as plt

#agent imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

#class imports
from Dueling_DoubleDQN_model import Dueling_DoubleDQN
from Environment_Snake import Environment_Snake
from Uniform_Experience_Replay import Uniform_Experience_Replay 

#exit handling
import signal
import sys

#use MPS
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('mps'))
print(f"\nDevice used: {device} \n")


#------------------ Helper functions ------------------

#handle user's exit from training loop
def signal_handler(sig, frame):
    global stop_training
    print("\nSoft shutdown initiated. Finishing current epoch...")
    stop_training = True


#save model params when called
def save_model_checkpoints(online_DDQN, target_DDQN, optimizer, episodes):
    
    #create state dictionaries
    DQN_state = {
    'frame': frame,
    'state_dict': online_DDQN.state_dict(),
    'optimizer': optimizer.state_dict(),
        }

    DQN_target_state = {
    'frame': frame,
    'state_dict': target_DDQN.state_dict(),
    'optimizer': optimizer.state_dict(),
        }

    print('... Saving DQN / Target DQN model parameters')
    DQN_filename = f'DQN_agent_episodes_{episodes}.pth.tar'
    DQN_target_filename = f'DQN_target_chkpt_episodes_{episodes}.pth.tar'

    torch.save(DQN_state,DQN_filename)
    torch.save(DQN_target_state,DQN_target_filename)
    print("... SAVED")


#plot training stats
def plot_training_statistics(running_rewards, 
                            running_qs_min, 
                            running_qs_max, 
                            foods_eaten, 
                            epsilons, 
                            n_foods, 
                            highscore):
    
    plt.style.use('dark_background')
    
    #Reward
    x_ = np.arange(0,np.array(running_rewards).shape[0])
    y = np.array(np.array(running_rewards))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='r',marker='+')
    plt.xlabel('Epochs (1000)')
    plt.ylabel('Average Reward')
    plt.title('Sliding-Window: Reward Training Epochs')

    #Qs
    x_ = np.arange(0,np.array(running_qs_min).shape[0])
    y = np.array(np.array(running_qs_min))
    #min
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='b',marker='+')
    x_ = np.arange(0,np.array(running_qs_max).shape[0])
    y = np.array(np.array(running_qs_max))
    #max
    plt.scatter(x_,y,s=4,c='r',marker='+')
    plt.xlabel('Epochs (1000)')
    plt.ylabel('Q Values')
    plt.title('Q values over Training Epochs')

    #foods
    x_ = np.arange(0,np.array(foods_eaten).shape[0])
    y = np.array(np.array(foods_eaten))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='b',marker='+')
    plt.xlabel('Game (every 100)')
    plt.ylabel('Foods Eaten Upon Death')
    plt.title('Foods Eaten Per Game')

    #epsilons
    x_ = np.arange(0,np.array(epsilons).shape[0])
    y = np.array(np.array(epsilons))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='m',marker='+')
    plt.xlabel('Epochs (1000))')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Values')

    #n_foods
    x_ = np.arange(0,np.array(n_foods).shape[0])
    y = np.array(np.array(n_foods))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='c',marker='+')
    plt.xlabel('Epochs (1000))')
    plt.ylabel('# of Food')
    plt.title('Number of Food During Training')

    #highscore
    x_ = np.arange(0,np.array(highscore).shape[0])
    y = np.array(np.array(highscore))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y,s=4,c='c',marker='+')
    plt.xlabel('Epochs (1000))')
    plt.ylabel('# of Food')
    plt.title('Highscore over Training')
    
if __name__ == "__main__":

    #init Environment and Agent params
    animate = True
    learning_rate = 0.001
    gamma = 0.99         
    epsilon_start = .75
    epsilon_min = 0.0001
    momentum = 0.00
    
    #training params
    frame =  0
    games_played =  0
    replay_buffer_length = 1_500_000
    batch_size = 128
    update_every = 8
    training_after = 10_000 + frame
    save_every_n_frames = 500_000
    sync_target_frames = 5000
    
    #how many frames deep the input to the model is, i.e. [8] x width x height
    frame_stack = 6

    #loss/reward metrics
    total_reward = deque([],maxlen=1000)
    total_loss = deque([0],maxlen=1000)
    
    #containers
    running_losses, running_qs_max, running_qs_min, running_rewards = [], [], [], []
    epsilons, n_foods, foods_eaten, highscore = [], [], [], []

    #state / experience
    state_space = deque([],maxlen=frame_stack)
    experience = namedtuple('Experience',['state_1',
                                          'state_2',
                                          'action',
                                          'delta_reward',
                                          'done_flag'])

    #initialize environment
    Environment = Environment_Snake(pygame,
                      width = 18,
                      height = 18,
                      cellsize = 25,
                      n_foods = 5,
                      fr = 0,
                      ticks = frame,
                      n_games = games_played,
                      animate=False
                     )
    
    Environment.initialize_board()
    Environment.initialize_game()

    #initialize agent and target model
    online_DDQN = Dueling_DoubleDQN(input_shape = [frame_stack,Environment.width+1,Environment.height+1],
                                n_actions = 4,
                                epsilon_start = epsilon_start,
                                epsilon_min = epsilon_min)
    
    target_DDQN = Dueling_DoubleDQN(input_shape = [frame_stack,Environment.width+1,
                                      Environment.height+1],
                                      n_actions = 4,
                                      epsilon_start = epsilon_start,
                                      epsilon_min = epsilon_min)    

    #RMSprop more stable for non-stationary
    optimizer = optim.RMSprop(online_DDQN.parameters(), 
                              lr = learning_rate, 
                              momentum = momentum)

    #initialize experience replay buffer 
    Buffer = Uniform_Experience_Replay(replay_buffer_length)

    #summary of model
    print(online_DDQN)
    print('\n')
    print(f'\n{summary(online_DDQN,(frame_stack,Environment.width+1,Environment.height+1))}\n')



    # ------------------ main play and training loop ------------------ #

    #exit logic
    stop_training = False
    # Register the signal handler ()
    signal.signal(signal.SIGINT, signal_handler)
   
   
    try:
        while not stop_training:     
            #add frame
            frame += 1
            
            #get game state @ s ^ t-1
            observation_pre = Environment.get_observation()
            
            #initialize pre and post framestack observations for Experience obj, clean this up ***
            pre_framestack = np.array(state_space)[0:frame_stack]
            
            #first few actions of game, need to fill up state space
            if frame <frame_stack+1:
                action = random.randint(1,4)
                Environment.snake_head['head'].append(action)
                current_dir = action
                delta_reward = 0
                state_space.append(observation_pre)
                Environment.update_environment(action,current_dir)

            #all actions after intro state-filling
            else:
                #stack state observations into singular input array (1 x framestack x height x width)
                state = np.rollaxis(np.dstack(state_space),-1)
                state_tensor = torch.from_numpy(state)[None]

                #first action after every death 
                if len(Environment.snake_head['head']) == 2:
                    action = torch.argmax(online_DDQN.forward(state_tensor)) + 1
                    Environment.snake_head['head'].append(action)
                    current_dir = action

                #if not first action, current dir = last action
                current_dir = Environment.snake_head['head'][2]   

                #get agent action --- mind the +1 to select action from index
                action = torch.argmax(online_DDQN.uniform_epsilon_greedy(frame,state_tensor)) + 1 
                Environment.snake_head['head'].append(action)

                #execute environment and receive reward, done flag status
                delta_reward, done_flag = Environment.update_environment(action,current_dir)
                
                #if current game is done
                if done_flag == True:
                    games_played += 1
                    if games_played % 100 == 0:
                        foods_eaten.append(Environment.death_foods)

                #get game state @ s ^ t=0
                observation_post = Environment.get_observation()

                #append post_obs to short memory deque
                state_space.append(observation_post)
                post_framestack = np.array(state_space)[:frame_stack+1]

                #Experience
                ex_ = experience(pre_framestack,post_framestack,action,delta_reward,done_flag)
                #Append experience to experience replay buffer
                Buffer.append(ex_)
            
            total_reward.append(delta_reward)

            #Save models
            if frame % save_every_n_frames == 0:
                save_model_checkpoints(online_DDQN, target_DDQN, optimizer, games_played)
                
            
            
            
            #reporting
            if frame % 10_000 == 0:
                #running reward, reporting, and training termination conditionals
                total_reward.append(delta_reward)
        
                #np functions don't play nice with the mps.tensor types...
                running_reward = (sum(total_reward) / len(total_reward))
                running_loss = (sum(total_loss) / len(total_loss))
                running_losses.append(running_loss)
                running_rewards.append(running_reward)
                running_qs_min.append((online_DDQN.forward(state_tensor)).detach().numpy().min())
                running_qs_max.append((online_DDQN.forward(state_tensor)).detach().numpy().max())
                highscore.append(Environment.session_highscore)
                epsilons.append(online_DDQN.epsilon)
                n_foods.append(Environment.n_foods)
                
                #print summary
                print(f"Frame: {frame} --- Games Played : {games_played} --- Running Reward : {running_reward}")
                print(f"Training completed in: {round(timeit.default_timer() - start_time)} seconds, device used: {device}")
                print(f"Epsilon: {online_DDQN.epsilon}")
                print(f"Running Loss: {running_loss}")
                print(f'Max q: {(online_DDQN.forward(state_tensor).max())} Min q: {(online_DDQN.forward(state_tensor).min())}')
                print(f"Session Highscore: {Environment.session_highscore}")
                print('----------------------------------------------------------------')
                print("Press ctrl+c to end training softly and save model parameters...\n\n")
                

            # ------------------ learning loop ------------------ #
            if frame % update_every == 0 and frame > training_after:
                
                #gather minibatch 
                states_1, states_2, actions, rewards, dones = Buffer.sample(batch_size)
                
                #unpack data --- in the case that a GPU is available, data will be loaded to GPU and graph computed there
                pre_states = torch.tensor(states_1).to(device)
                post_states = torch.tensor(states_2).to(device)
                
                #subtract 1 to restore the appropriate index 
                actions = torch.tensor(actions).to(device) - 1
                rewards = torch.tensor(rewards).to(device)
                dones = torch.ByteTensor(dones).to(device).bool()

                #seems essential to load these models to MPS, to maintain correct tensor type
                online_DDQN = online_DDQN.to(device)
                target_DDQN = target_DDQN.to(device)

                #inspired by Maxim Lapan
                #https://tinyurl.com/2s44tepw

                #   1. 
                #   We show the Agent DQN its past states, and capture it's corresponding q-values for each possible
                #   Action. We gather up the q-values for actions it took using .gather and our 'actions' vector
                #   This is effectively our prediction or x 
                online_Q_values = online_DDQN.forward(pre_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                
                #   2. 
                #   We need our target q-values for the same state/action pairs we just gathered
                #   We use the target DQN to avoid a shifting target when backpropagating error gradient
                #   For DDQN, we evaluate the greedy-policy with our online network and then use the selected action to estimate Qs-
                #   using our target network 
            
                #capture post_state actions --- no need to restore appropriate index as argmax return index
                online_actions = torch.argmax(online_DDQN.forward(post_states),dim=1)
                
                #use online actions as indices to target network's forward pass, generating post state-action values
                target_Q_values = target_DDQN.forward(post_states).gather(1, online_actions.unsqueeze(-1)).squeeze(-1)
                
                #Where the done flag is True, we have post_state_values = 0
                #There is no next state, as episode has ended 
                target_Q_values[dones] = 0.0
                
                #detach from torch graph so it doesn't mess up gradient calculation
                target_Q_values = target_Q_values.detach()
                
                #   3.
                #   calculate expected state-actiom values using the bellman equation 
                #   this is effectively our 'y' for our mean-squared error loss function
               
                target_Q_values = target_Q_values * gamma + rewards
                
                #   4. 
                #   calculate Huber loss - less sensitive to outliers - effectively the Temporal-Difference loss
                loss = nn.HuberLoss()(online_Q_values , target_Q_values)
                
                #   5.
                #   backpropagate error of loss function
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.detach())

                #seems essential to load these models back to CPU, for gameplay outside of training loop. 
                online_DDQN = online_DDQN.to('cpu')
                

            #iteratvely update target net values to better estimate target values
            if frame % sync_target_frames == 0 and frame > training_after + 1000:
                target_DDQN.load_state_dict(online_DDQN.state_dict())
    

            #final board draw, if animate = True
            Environment.draw_board()   

    #error handling
    except Exception as e:
            print(f"Error occurred: {e}")

    finally:
        plot_training_statistics(running_rewards, running_qs_min, running_qs_max, foods_eaten, epsilons, n_foods, highscore)
        #save_model_checkpoints(online_DDQN, target_DDQN, optimizer, games_played)
        
        print(f"Training loop has ended. Cleaning up resources...",
               "\nFinal Model parameters saved... ")

        #final 
        #plt.show()
        pygame.quit()
    
              

    













