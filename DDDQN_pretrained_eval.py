#environment imports
import numpy as np
import pygame 
import random

#memory replay data structures
from collections import deque, namedtuple

#plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#agent imports
import torch
import torch.nn as nn
import torch.optim as optim

#old numpy warning
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#use CPU
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"\nDevice used: {device} \n")

class Dueling_DoubleDQN(nn.Module):
    '''

    '''

    def __init__(self, input_shape, n_actions):
        super().__init__()


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

    #probabilistically generates a random action, so model doesn't get stuck
    def random_action(self, observation):
        p_random = 5000
        random =  np.random.randint(1, p_random+1)
        if p_random == random:
            self.action = np.random.uniform(-1,0,4)
            return torch.tensor(self.action)
        else:
            return self.forward(observation)


#Evironment class, contains all methods to render and play snake 
class Env:
    def __init__(self,pygame,width,height,cellsize,n_foods,fr,death_delay):
        #game elements
        self.pygame = pygame
        self.width = width
        self.height = height
        self.cellsize = cellsize
        self.fr = fr
        self.death_delay = death_delay
        self.n_foods = n_foods
       
        #game stats
        self.session_highscore = 0
        self.n_games = 1
        self.ticks = 0 
        self.gamescores = []

    #init pygame surface / pygame
    def initialize_board(self):
        #initialize pygame and surface animation
        self.pygame.init() 
        self.surface = pygame.display.set_mode(((self.width+1)*self.cellsize,
                                                (self.height+1)*self.cellsize)) 
        
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

        #score and game info
        
        self.draw_board()

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
                self.record_games(self.segments)
                self.done = True
                pygame.time.delay(self.death_delay)
                self.initialize_game()
                delta_reward = delta_reward - 40                                                   #20 -> 40
                self.n_games += 1
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
        #iterate through every cell in grid 
        for r, c in np.ndindex(self.width+1,self.height+1): #+1 for wall pads
           
            if self.board[r,c] != 4:
                if self.board[r,c] == 1: #head
                    col = (220,20,60)
                elif self.board[r,c] == 2: #body
                    col = (65, 105 + random.randint(0,30), 225 + random.randint(0,30))
                elif self.board[r,c] == 3: #food
                    col = (143,196,54)
                elif self.board[r,c] == 0:
                    col = (22, 18, 64)
                self.pygame.draw.rect(self.surface, col, (r*self.cellsize, c*self.cellsize, 
                                                                self.cellsize-1, self.cellsize-1)) #draw new cell
            else: #wall
                col = (0,139,139)
                self.pygame.draw.rect(self.surface, col,(r*self.cellsize, c*self.cellsize, 
                                                                self.cellsize, self.cellsize)) #draw new cell

        pygame.display.update() #updates display from new .draw in update function


    #spawn food
    def spawn_food(self):
        #spawn food in a loop until not
        while True:
            not_overlapping = True
            self.food = list(tuple((random.randint(2,self.width-1),
                                random.randint(2,self.height-1)) for _ in range(self.n_foods)))
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
                new_food = tuple((random.randint(2,self.width-1),
                                  random.randint(2,self.height-1)))
                #overlapping?
                if self.board[new_food[0],new_food[1]] == 0:
                    break
        self.food += [new_food]
        

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

        self.pygame.display.set_caption(f" 'BOT' --- SCORE: {self.segments}  -  SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")


    #REPORT OBSERVATION
    def get_observation(self):
        #capture state as np array, convert to tensor for torch
        state = np.transpose(self.board)
        state = state.reshape(1,1,self.height+1,self.width+1)
        state = torch.from_numpy(state)
        
        #normalization of pixel values,  max value is 3
        state = state.float() / 4
        return state.cpu().detach().numpy().squeeze()
    
    def record_games(self,gamescore):
        self.gamescores.append(gamescore)


#plotting function for evaluation
def plotting(gamescores,Environment):
    gamescores.sort()
    x_ = np.arange(0,np.array(gamescores).shape[0])
    y_ = np.array(np.array(gamescores))
    plt.figure(figsize=(12,7))
    plt.scatter(x_,y_,s=4,c='g',marker='+')
    plt.xlabel("")
    plt.ylabel('Sorted Game Scores')
    plt.xlabel('Games')
    plt.title(f'Model Evaluation --- Model Best: {round(((max(y_)) / (Environment.width*Environment.height)*100),3)}% Board Completion --- {Environment.n_games} Games --- average score: {np.mean(gamescores)}')
    plt.axhline(y=max(y_), color='r', linestyle='--')
    #plt.show()





if __name__ == "__main__":

    #init Environment and Agent params
    frame = 0
    #how many frames to stack for model input
    frame_stack = 6
    state_space = deque([],maxlen=frame_stack)

    #initialize environment 
    Environment = Env(pygame,
                      width = 16,
                      height = 16,
                      cellsize = 40,
                      n_foods = 1,
                      fr = 80,
                      death_delay = 400
                     )
    
    Environment.initialize_board()
    Environment.initialize_game()
    n_evaluations = 1000
    

    DQN_ = Dueling_DoubleDQN(input_shape = [frame_stack,Environment.width+1,
                                Environment.height+1],
                                n_actions = 4,
                                )
    

    #eval -------------------------------------------------- 

    #load path
    agent_training_checkpoint = torch.load('--.TAR MODEL PATH HERE--')
    DQN_.load_state_dict(agent_training_checkpoint['state_dict'])
    DQN_.eval()
                 
    print(f'\n{DQN_}\n')

    # ------------------ main play and eval ------------------ #

    while Environment.n_games < n_evaluations:     
        for event in Environment.pygame.event.get(): 
                Event = event
                #exit window
                if Event.type == pygame.QUIT:
                    pygame.quit()
                    #evaluation plotting
                    print(Environment.gamescores)
                    plotting(Environment.gamescores,Environment)
                    exit()

        frame +=1

        #first couple frames
        if frame < frame_stack+1:
            #get game state @ s ^ t-1
            observation_pre = Environment.get_observation()
            action = random.randint(1,4)
            Environment.snake_head['head'].append(action)
            current_dir = action
            delta_reward = 0
            state_space.append(observation_pre)
            Environment.update_environment(action,current_dir)

        else:

            #delay for animation framerate 
            pygame.time.delay(Environment.fr)

            #stack state observations into singular input array (1 x 4 x height x width)
            state = np.rollaxis(np.dstack(state_space),-1)
            state_tensor = torch.from_numpy(state)[None]

            #get agent action --- mind the +1 to select action from index
            action = torch.argmax(DQN_.random_action(state_tensor)) + 1 
            Environment.snake_head['head'].append(action)
            #if not first action, current dir = last action
            current_dir = Environment.snake_head['head'][2]   
            #execute environment and receive reward, done flag status
            delta_reward, done_flag = Environment.update_environment(action,current_dir)
            #get game state @ s ^ t=0
            observation_post = Environment.get_observation()
            #append post_obs to short memory deque
            state_space.append(observation_post)
            #final board draw
            Environment.draw_board() 
    
    plotting(Environment.gamescores,Environment)
    





            


            
    

    
    













        

























