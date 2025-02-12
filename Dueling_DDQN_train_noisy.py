#environment imports
import numpy as np
import pygame 
import random
import timeit
start_time = timeit.default_timer()

#memory replay data structures
from collections import deque, namedtuple
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#agent imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from copy import deepcopy

#class imports
from Noisy_Dueling_DoubleDQN_model import Noisy_Dueling_DoubleDQN, NoisyLinear
from Environment_Snake import Environment_Snake
from Prioritized_Experience_Replay import SumTree, PrioritizedExperienceReplay
from Uniform_Experience_Replay import Uniform_Experience_Replay
from TrainingStatisticsPlotter import TrainingStatisticsPlotter

#exit handling
import signal
import sys

#use MPS
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

    
if __name__ == "__main__":

    #init Environment and Agent params
    animate = False
    learning_rate = 0.00025
    gamma = 0.96 
    momentum = 0.00
    
    #'Uniform' or 'Prioritized' replay buffer
    buffer_type = 'Prioritized'

    #training params
    frame =  0
    games_played =  0
    replay_buffer_length = 800_000
    batch_size = 128
    update_every = 8
    training_after = 10_000 + frame
    save_every_n_frames = 200_000
    sync_target_frames = 5000
    report_every_n_frames = 1000
    plot_distributions_frames = 30_000
    

    #how many frames deep the input to the model is, e.g. [8] x width x height
    frame_stack = 6

    #loss/reward metrics
    total_reward = deque([],maxlen=5000)
    total_loss = deque([0],maxlen=5000)
    PER_priorities = deque([],maxlen=300)
    init_priority = 1 #initialize first priorities to 1, before first init priority scheduler
    
    #containers
    running_losses, running_qs_max, running_qs_min, running_rewards = [], [], [], []
    epsilons, n_foods, foods_eaten, highscore, PER_beta, layer_noise = [], [], [], [], [], []

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
                      animate=animate
                     )
    
    Environment.initialize_board()
    Environment.initialize_game()

    #initialize agent and target model
    online_DDQN = Noisy_Dueling_DoubleDQN(input_shape = [frame_stack,Environment.width+1,Environment.height+1],
                                n_actions = 4)
    
   
    print(f"online_DDQN.device: {next(online_DDQN.parameters()).device}")
    
    target_DDQN = deepcopy(online_DDQN)

    optimizer = optim.Adam(online_DDQN.parameters(), 
                       lr=0.00025,  # Learning rate
                       betas=(0.9, 0.999),  # Standard for RL
                       eps=1e-8)  # Avoids numerical instability
    
    #visualize training metrics
    plotter = TrainingStatisticsPlotter()

    #initialize experience replay buffer 
    if buffer_type == 'Prioritized':
        Buffer = PrioritizedExperienceReplay(replay_buffer_length)
    elif buffer_type == 'Uniform':
        Buffer = Uniform_Experience_Replay(replay_buffer_length)

    #summary of model
    print(online_DDQN)
    print('\n')
    print(f'\n{summary(online_DDQN,(frame_stack,Environment.width+1,Environment.height+1))}\n')
    print(f'\nReplay Buffer type: {buffer_type}\n\n\n')


    # ------------------ main play and training loop ------------------ #

    #exit logic
    stop_training = False
    # Register the signal handler ()
    signal.signal(signal.SIGINT, signal_handler)
   
   
    try:
        while not stop_training:     
            
            #seems essential to load these models back to CPU, for gameplay outside of training loop. 
            online_DDQN = online_DDQN.to('cpu')
            
            #add frame
            frame += 1
            
            #get game state @ s ^ t-1
            observation_pre = Environment.get_observation()
            
            #initialize pre and post framestack observations for Experience object
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
                action = torch.argmax(online_DDQN.forward(state_tensor)) + 1 
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
                
                #if using Uniform Experience Replay, just append experience alone at this step
                if isinstance(Buffer, Uniform_Experience_Replay):
                    #Append experience to experience replay buffer
                    Buffer.append(ex_)
                    #gather minibatch 
                    states_1, states_2, actions, rewards, dones = Buffer.sample(batch_size)
                
                #if using Prioritized Experience Replay, initialize add with delta reward
                elif isinstance(Buffer, PrioritizedExperienceReplay):
                    #initialize priorities as the maximum priority in the tree, ensuring sampling at least once
                    Buffer.add(ex_,init_priority) #ex_ without TD means priority default is maximum_priority in Buffer.tree
                    # Sample a minibatch
                    states_1, states_2, actions, rewards, dones, is_weights, idxs = Buffer.sample(batch_size)
                    
                    #record PER sampling priorities for sampled transitions
                    priorities = Buffer.tree.tree[idxs]  # Leaf node priorities
                    PER_priorities.append(priorities)


            #append reward signal to total_reward, in order to calculate running reward during training
            total_reward.append(delta_reward)


            # ------------------ learning loop ------------------ #
            if frame % update_every == 0 and frame > training_after:
               
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
                #   We use the target DQN to avoid a shifting target when backpropagating error gradient (Double DQN)
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
                #   calculate expected state-action values using the bellman equation 
                #   this is effectively our 'y' for our mean-squared error loss function, and incorporates feedback from the environment
                target_Q_values = target_Q_values * gamma + rewards
                
                #Handle loss according to chosen replay method
                #   4. 
                #   calculate Huber loss - less sensitive to outliers - effectively the Temporal-Difference loss
                if isinstance(Buffer, Uniform_Experience_Replay):
                    loss = nn.HuberLoss()(online_Q_values , target_Q_values)
                
                #   Huber loss, with importance sampling correction (see paper)
                elif isinstance(Buffer, PrioritizedExperienceReplay):
                    # Compute loss with importance-sampling weights
                    is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32).to(device)
                    loss = (nn.HuberLoss(reduction="none")(online_Q_values, target_Q_values) * is_weights_tensor).mean()
                
                #   5.
                #   backpropagate error of loss function
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss.append(loss.detach())
        
                #   6.
                #   if using PER, update the priorities of experiences based on computed TD:
                if isinstance(Buffer, PrioritizedExperienceReplay):
                    # Update priorities in the buffer
                    td_errors = torch.abs(online_Q_values - target_Q_values).detach().cpu().numpy()
                    Buffer.update_priorities(idxs, td_errors)
        


            # ------------------ iterative training logic ------------------ #
            #iteratvely update target net values to better estimate target values
            if frame % sync_target_frames == 0 and frame > training_after + 1000:
                target_DDQN.load_state_dict(online_DDQN.state_dict())

           

            #Save models
            if frame % save_every_n_frames == 0:
                save_model_checkpoints(online_DDQN, target_DDQN, optimizer, games_played)

            #save the sampled priority distributions to track PER evolution
            if frame % plot_distributions_frames == 0:
                #Visualize the sampling distribution for the PER
                plotter.sampled_priority_distributions(PER_priorities,frame,Buffer)
                plotter.plot_all(frame, running_rewards, running_qs_min, running_qs_max, foods_eaten, epsilons, n_foods, highscore, PER_beta)




            
            #dynamically adjust Buffer max_priority for new experience init
            if frame % Buffer.recalculate_schedule == 0:
                #flatten 
                recent_priorities = [priority for sublist in PER_priorities for priority in sublist]
                window_mean_priority = sum(recent_priorities) / len(recent_priorities)
                init_priority = window_mean_priority
                print(f"New Init Priority: {init_priority}")
            

            #training progress reporting
            if frame % report_every_n_frames == 0:
                total_reward.append(delta_reward)
                #np functions don't play nice with the mps.tensor types...
                running_reward = (sum(total_reward) / len(total_reward))
                running_loss = (sum(total_loss) / len(total_loss))
                running_losses.append(running_loss)
                running_rewards.append(running_reward)
                
                '''
                this is broken
                #running_qs_min.append((online_DDQN.forward(state_tensor)).detach().numpy().min())
                #running_qs_max.append((online_DDQN.forward(state_tensor)).detach().numpy().max())
                '''
                highscore.append(Environment.session_highscore)
                n_foods.append(Environment.n_foods)
                PER_beta.append(Buffer.beta)

                #track PER buffer sampling distribution
                mean_priority = np.mean(PER_priorities[-1])
                max_priority = np.max(PER_priorities[-1])
                min_priority = np.min(PER_priorities[-1])
                
                #print summary
                print(f"\n\n\nFrame: {frame} --- Games Played : {games_played} --- Running Reward : {running_reward:.4f}")
                print(f"Training completed in: {round(timeit.default_timer() - start_time)} seconds, device used: {device}")
                print(f"Running Loss: {running_loss}")
                '''
                this is broken
                print(f'Max q: {(online_DDQN.forward(state_tensor).max())} Min q: {(online_DDQN.forward(state_tensor).min())}')
                '''
                print(f"PER Sampling Priorities in most recent Sample: - Mean: {mean_priority:.4f}, Max: {max_priority:.4f}, Min: {min_priority:.4f}, Buffer Max Priority: {Buffer.tree.max_priority:.4f}")
                print(f"Session Highscore: {Environment.session_highscore}")
                #track noise in model dense layers
               
                for name, layer in online_DDQN.to('cpu').named_modules():
                    if isinstance(layer, NoisyLinear):
                        weight_noise, bias_noise = layer.get_noise_level()
                        print(f"Layer {name} - Weight Noise: {weight_noise:.6f}, Bias Noise: {bias_noise:.6f}")
                print('----------------------------------------------------------------')
                print("Press ctrl+c to end training softly and save model parameters...")
    
            
            #final board draw, if animate = True
            Environment.draw_board()   

    #error handling
    except Exception as e:
            print(f"Error occurred: {e}")

    finally:
        #Plot training stats
        plotter.plot_all(running_rewards, running_qs_min, running_qs_max, foods_eaten, epsilons, n_foods, highscore, PER_beta)
        #save_model_checkpoints(online_DDQN, target_DDQN, optimizer, games_played)
        plotter.sampled_priority_distributions(PER_priorities,frame,Buffer)
        
        print(f"Training loop has ended. Cleaning up resources...",
               "\nFinal Model parameters saved... ")

        #matplotlib and pygame spindown
        #plt.show()
        pygame.quit()
    









