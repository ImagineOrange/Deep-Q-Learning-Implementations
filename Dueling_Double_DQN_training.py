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

#old numpy warning
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#use MPS
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('mps'))
print(f"\nDevice used: {device} \n")



class Dueling_DoubleDQN(nn.Module):
    '''

    '''

    def __init__(self, input_shape, n_actions, epsilon_start, epsilon_min):
        super().__init__()

        #epsilon params
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_start = epsilon_start

        #nn.Conv2d modules expect a 4-dimensional tensor with the shape [batch_size, channels, height, width] 
        
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),  # Output: 20x20
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
        )


        self.advantage_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape),128),
            nn.ReLU(),
            nn.Linear(128,n_actions),
            
       )

        self.value_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape),128),
            nn.ReLU(),
            nn.Linear(128,1),
       )

    

    def get_conv_out(self, input_shape):
         o = self.conv_features(torch.zeros(1, * input_shape))
         o = int(np.prod(o.size()))
         return o


    def forward(self, x):
        #get features from convolutional layer
        conv_out = self.conv_features(x).view(x.size()[0], -1)
        #value from value stream
        value = self.value_stream(conv_out)
        #advantages from advantage stream
        actions_advantages = self.advantage_stream(conv_out)
        #average advantage
        average_advantages  = torch.mean(actions_advantages, dim=1, keepdim=True)

        Q = value + (actions_advantages - average_advantages)

        #return Q values
        return Q

    
    def uniform_epsilon_greedy(self,frame,observation):
        self.frame = frame
        self.current_dir = current_dir
        

        #anneal epsilon over 1.5 million frames
        if self.epsilon > self.epsilon_min:
           self.epsilon = self.epsilon_start - .00000067 * frame
        
        #pick a random number from uniform distribution
        self.rand = random.uniform(0, 1)
        #if epsilon larger, take random action
        if self.rand <= self.epsilon:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action)
        #else, run forward pass on observation
        return self.forward(observation)







class Uniform_Experience_Replay:
    '''
    
    '''

    def __init__(self, capacity):
        #initialize deque of maxlenlen capacity
        self.buffer = deque(maxlen=capacity)
    
    def append(self, experience):
        #append experience stack to deque
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        #random indices of deque
        indices = np.random.default_rng().choice(len(self.buffer), batch_size, replace=False)
        #the zip + * op is going to iterate over all data types in the deque at our indices
        states_1, states_2, actions, rewards, dones = zip(*[self.buffer[idx] for idx in indices])
        #return np arrays of unpacked data
        return np.array(states_1), np.array(states_2), np.array(actions), np.array(rewards), np.array(dones)







class Prioritized_Experience_Replay:
   pass






#Evironment class, contains all methods to render and play snake 
class Environment_Snake:
    '''

    '''

    def __init__(self,pygame,width,height,cellsize,n_foods,fr,ticks,n_games):
        #game elements
        self.pygame = pygame
        self.width = width
        self.height = height
        self.cellsize = cellsize
        self.fr = fr
        self.n_foods = n_foods
       
        #game stats
        self.session_highscore = 0
        self.n_games = n_games
        self.ticks = ticks

        #remove foods periodically, helps with early training to have more than 1 food
        self.remove_schedule = [100_000]
        for i in range(1,n_foods):
            self.remove_schedule.append(self.remove_schedule[-1] + 100_000)
       
        print(f"Number of starting foods: {n_foods}\nRemove Schedule: {self.remove_schedule}\n")

    #init pygame surface / pygame
    def initialize_board(self):
        #initialize pygame and surface animation
        self.pygame.init() 
        #self.surface = pygame.display.set_mode(((self.width+1)*self.cellsize,
        #                                        (self.height+1)*self.cellsize)) 
        #self.pygame.display.set_caption(' ')
    
        
    #init / reset board
    def initialize_game(self):
        #pad for border of 1
        self.board = np.pad(np.zeros([self.width-1,self.height-1]),
                            pad_width=(1,1),mode='constant',constant_values = 4)
        
        #snake head dict contains head coords, and also current direction
        self.snake_head = {
                            'head' : ([random.randint(5,self.width-2),
                                       random.randint(5,self.height-2)])
                          }

        #snake body, empty on init
        self.snake_body = {}
        
        #number of segments, 0 on init
        self.segments = 0
        
        #draw head on board
        self.board[self.snake_head.get('head')[0],
                   self.snake_head.get('head')[1]] = 1

        #self.pygame.display.set_caption(f" 'Dueling DDQN' --- SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")

        #spawn initial food 
        self.spawn_food()

    #update head location on board
    def update_head_location(self,current_dir,action):
        #negate illegal moves, can't go backwards 
        if current_dir == 1 and action == 2 or \
           current_dir == 2 and action == 1 or \
           current_dir == 3 and action == 4 or \
           current_dir == 4 and action == 3 :
           action = current_dir

        #conditionals for actions
        if action == 1:
            old_hed_pos = self.snake_head.get('head')                    #[x coordinate, y coordinate, direction]
            self.new_hed_pos = [old_hed_pos[0],old_hed_pos[1]-1,1]
        elif action == 2:
            old_hed_pos = self.snake_head.get('head')
            self.new_hed_pos = [old_hed_pos[0],old_hed_pos[1]+1,2]
        elif action == 3:
            old_hed_pos = self.snake_head.get('head')
            self.new_hed_pos = [old_hed_pos[0]+1,old_hed_pos[1],3]
        elif action == 4:
            old_hed_pos = self.snake_head.get('head')
            self.new_hed_pos = [old_hed_pos[0]-1,old_hed_pos[1],4]

    #update segment location on board
    def update_segment_locations(self):
        #update coords of body segments
        self.old_body_pos = self.snake_body.copy()
        for segment in self.snake_body:
            #if neck
            if segment == '1':
                new_values = self.snake_head.get('head')
                self.snake_body.update({'1': new_values})
                #draw snake on board
                self.board[self.snake_body.get('1')[0],self.snake_body.get('1')[1]] = 2
            #if body
            else:
                key = f'{(int(segment)-1)}'
                new_values = self.old_body_pos.get(key)
                self.snake_body.update({segment: new_values})
                #draw segment on board
                self.board[self.snake_body.get(segment)[0],self.snake_body.get(segment)[1]] = 2

    #update board according to last action, action, and rules
    def update_environment(self,action,current_dir):
        self.ticks += 1
        delta_reward = 0
        #update head location
        self.update_head_location(current_dir,action)
    
        #head hits wall or self
        if self.board[self.new_hed_pos[0],self.new_hed_pos[1]] == 2 \
            or self.board[self.new_hed_pos[0],self.new_hed_pos[1]] == 4:
                self.done = True
                self.death_foods = self.segments
                self.reduce_food_n()
                self.initialize_game()
                delta_reward = delta_reward - 40                                                   
                self.n_games +=1
                return delta_reward, self.done
            
        #if get food
        if self.board[self.new_hed_pos[0],self.new_hed_pos[1]] == 3 :
           self.done = False
           self.grow_snake()
           self.update_environment(action,current_dir)
           delta_reward += 10 + (self.segments * 4)
           return  delta_reward, self.done


        #all legal moves    
        else:
            #reset board array for animation
            self.board = np.pad(np.zeros([self.width-1,self.height-1]),
                                pad_width=(1,1),mode='constant',constant_values = 4)
            #keep food
            for item in self.food:
                self.board[(item[0]),(item[1])] = 3
            #update segment locations
            self.update_segment_locations()
            #update head coordinates
            self.snake_head.update({'head': self.new_hed_pos})
            #draw snake head 
            self.board[self.snake_head.get('head')[0],self.snake_head.get('head')[1]] = 1
            self.done = False
            return delta_reward, self.done

    #draw board
    def draw_board(self):
        '''
        #iterate through every cell in grid 
        for r, c in np.ndindex(self.width+1,self.height+1): #+1 for wall pads
            if self.board[r,c] != 4:
                if self.board[r,c] == 1: #head
                    col = (220,20,60)
                elif self.board[r,c] == 2: #body
                    col = (255, 80 + random.randint(0,30), 3 + random.randint(0,30))
                elif self.board[r,c] == 3: #food
                    col = (111, 153, 64)
                elif self.board[r,c] == 0:
                    col = (22, 18, 64)
                self.pygame.draw.rect(self.surface, col,(r*self.cellsize, c*self.cellsize, 
                                                           self.cellsize-.9, self.cellsize-.9)) #draw new cell
            else: #wall
                col = (0,139,139)
                self.pygame.draw.rect(self.surface, col,(r*self.cellsize, c*self.cellsize, 
                                                                self.cellsize, self.cellsize)) #draw new cell
        pygame.display.update() #updates display from new .draw in update function
        '''

    #spawn food
    def spawn_food(self):
        #spawn food in a loop until not
        while True:
            not_overlapping = True
            self.food = list(tuple((random.randint(1,self.width),
                                random.randint(1,self.height)) for _ in range(self.n_foods)))
            #check if overlapping
            for item in self.food:
                if self.board[item[0],item[1]] != 0:
                    not_overlapping = False
               
            if not_overlapping == True:
                for item in self.food:
                    self.board[item[0],item[1]] = 3
                break
    
    #add set of new food indices to self.food index list
    def add_food(self):
        #place head at old food spot
        self.board[self.new_hed_pos[0],self.new_hed_pos[1]] = 1 
        #delete just-eaten food
        self.food.remove(tuple([self.new_hed_pos[0],self.new_hed_pos[1]]))
        #loop until new food is non-overlapping with other game tiles
        while True:
            #add new food at index not occupied by snake or walls
            if len(self.food) < self.n_foods:
                new_food = tuple((random.randint(1,self.width),
                                  random.randint(1,self.height)))
                #overlapping?
                if self.board[new_food[0],new_food[1]] == 0:
                    break
        self.food += [new_food]
        
    #reduce food during training -- notice gameticks instead of frames here
    def reduce_food_n(self):
        if self.done:
            if self.ticks > self.remove_schedule[0] and self.n_foods > 1:
                self.n_foods -= 1
                print(f'Number of Foods: {self.n_foods}')
                del self.remove_schedule[0] 

    #GROW SNAKE WHEN EAT FOOD
    def grow_snake(self):
        #snake head dict contains head coords, and also current direction
        self.segments +=1 
        #add 1 food
        self.add_food()
        
        if self.segments == 1:
            self.snake_body[f'{self.segments}'] = [self.snake_head.get('head')[0],
                                                   self.snake_head.get('head')[1]+1,
                                                   self.snake_head.get('head')[2]]        

        #add new segment
        else:
            self.snake_body[f'{self.segments}'] = self.snake_body.get(f'{self.segments-1}')

        if self.segments > self.session_highscore:
                self.session_highscore = self.segments

        #self.pygame.display.set_caption(f" 'Dueling DDQN' --- SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")

    #REPORT OBSERVATION
    def get_observation(self):
        #capture state as np array, convert to tensor for torch
        state = np.transpose(self.board)
        state = state.reshape(1,1,self.height+1,self.width+1)
        state = torch.from_numpy(state)
        
        #normalization of pixel values,  max value is 3
        state = state.float() / 4
        return state.detach().numpy().squeeze()

def save_model_checkpoints(DQN_state,
                           DQN_target_state,
                           episodes):
    
    print('... Saving DQN / Target DQN model parameters')
    DQN_filename = f'CHEEMS_DQN_agent_episodes_{episodes}.pth.tar'
    DQN_target_filename = f'CHEEMS_DQN_target_chkpt_episodes_{episodes}.pth.tar'

    torch.save(DQN_state,DQN_filename)
    torch.save(DQN_target_state,DQN_target_filename)
    print("... SAVED")


'''
############################################################
  _             _       _               _                   
 | |           (_)     (_)             | |                  
 | |_ _ __ __ _ _ _ __  _ _ __   __ _  | | ___   ___  _ __  
 | __| '__/ _` | | '_ \| | '_ \ / _` | | |/ _ \ / _ \| '_ \ 
 | |_| | | (_| | | | | | | | | | (_| | | | (_) | (_) | |_) |
  \__|_|  \__,_|_|_| |_|_|_| |_|\__, | |_|\___/ \___/| .__/ 
                                 __/ |               | |    
                                |___/                |_|    
############################################################
'''


if __name__ == "__main__":

    #init Environment and Agent params
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
    running_losses = []
    running_qs_max = []
    running_qs_min = []
    running_rewards = []
    epsilons = []
    n_foods = []
    foods_eaten = []
    highscore = []

    #state / experience
    state_space = deque([],maxlen=frame_stack)
    experience = namedtuple('Experience',['state_1',
                                          'state_2',
                                          'action',
                                          'delta_reward',
                                          'done_flag'])

    #initialize environment 
    Environment = Environment_Snake(pygame,
                      width = 16,
                      height = 16,
                      cellsize = 70,
                      n_foods = 5,
                      fr = 0,
                      ticks = frame,
                      n_games = games_played
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
   


   
    RUNNING = True
    try:
        while RUNNING:     
            for event in Environment.pygame.event.get(): 
                #exit window
                if event.type == pygame.QUIT:
                    # --------------------------- final plotting------------------------------- #
                    plt.style.use('dark_background')

                    #Reward
                    x_ = np.arange(0,np.array(running_rewards).shape[0])
                    y = np.array(np.array(running_rewards))

                    plt.figure(figsize=(12,7))
                    plt.scatter(x_,y,s=4,c='r',marker='+')
                    plt.xlabel('Epochs (1000)')
                    plt.ylabel('Average Reward')
                    plt.title('Sliding-Window: Reward Training Epochs')

                    '''
                    #Loss
                    x_ = np.arange(0,np.array(running_losses).shape[0])
                    y = np.array(np.array(running_losses))

                    plt.figure(figsize=(12,7))
                    plt.scatter(x_,y,s=4,c='g',marker='+')
                    plt.xlabel('Epochs (1000)')
                    plt.ylabel('Average Losses')
                    plt.title('Sliding-Window: Loss - Training Epochs')
                    '''
                
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
                    
                    RUNNING = False
                    # --------------------------- final plotting------------------------------- #

            #if not quit...
            frame += 1
            #Save models
            if frame % save_every_n_frames == 0:
                
                DDQN_state = {
                'frame': frame,
                'state_dict': online_DDQN.state_dict(),
                'optimizer': optimizer.state_dict(),
                 }

                DDQN_target_state = {
                'frame': frame,
                'state_dict': target_DDQN.state_dict(),
                'optimizer': optimizer.state_dict(),
                 }

                save_model_checkpoints(DDQN_state,DDQN_target_state,games_played)
            

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
            if frame % 1000 == 0:
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
                print(f"Training completed in: {timeit.default_timer() - start_time} seconds, device used: {device}")
                print(f"Epsilon: {online_DDQN.epsilon}")
                print(f"Running Loss: {running_loss}")
                print(f'Max q: {(online_DDQN.forward(state_tensor)).max()} Min q: {(online_DDQN.forward(state_tensor)).min()} \n')
                

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
    
            #final board draw
            #Environment.draw_board()   

    finally:
        plt.show()
        pygame.quit()



'''
currently configured to not animate the snake!
testing training variables!
testing Metal Performance Shaders (MPS)

jan 7th experiment: 
    -longer replay buffer (1_500_000)
    -larger batch size (64)
    -MPS compute
    -larger base epsilon - .75
    -sync target frames every 5000 - stability?
    -update every 8 frame - stability?
    -framestack now 6 not 4 --- more motion inference?
'''










