#class imports
import pygame
import numpy as np
import torch
import random 

#Evironment class, contains all methods to render and play snake 
class Environment_Snake:
    '''
    Game environment for Reinforcement Learning Experiments
    Args:
        pygame   : pygame
        width    : playable gameboard width 
        height   : playable gameboard height
        cellsize : size for each animated game pixel
        n_foods  : number of foods to place in game environment, if n_foods > 0 one food is 
                  removed every 100_000 frames until n_foods = 1
        fr       : animation framerate **note** training much slower with animation enabled, to 
                   disable, comment out pygame board code...will add flag later (1/11/25)
        ticks    : game ticks (frames played, default 0)
        n_games  : number of games played (default 0)

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
        self.surface = pygame.display.set_mode(((self.width+1)*self.cellsize,
                                                (self.height+1)*self.cellsize)) 
        self.pygame.display.set_caption(' ')
    
        
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

        self.pygame.display.set_caption(f"SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")

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

        #self.pygame.display.set_caption(f"SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")

    #REPORT OBSERVATION
    def get_observation(self):
        # Capture state as np array and convert to tensor for torch
        state = np.transpose(self.board)
        state = state.reshape(1, 1, self.height + 1, self.width + 1)
        state = torch.from_numpy(state)
        
        # Add a small constant to each pixel
        constant = 0.25  #no dead neurons/vanishing gradients
        state = state.float() + constant
        
        # Normalization of pixel values, max value is 3
        state = state / 4
        
        # Return the state as a numpy array
        return state.cpu().detach().numpy().squeeze()
