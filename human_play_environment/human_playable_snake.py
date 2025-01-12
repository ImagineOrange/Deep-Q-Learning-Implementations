#environment imports
import numpy as np
import pygame
from pygame.locals import *
import random



#Evironment class, contains all methods to render and play snake 
class Environment_Snake:
    def __init__(self,pygame,width,height,cellsize,fr,n_games,n_foods,game_ticks):
        #game elements
        self.pygame = pygame
        self.width = width
        self.height = height
        self.cellsize = cellsize
        self.fr = fr
       
        #game stats
        self.session_highscore = 0
        self.n_games = n_games
        self.n_foods = n_foods
        self.game_ticks = game_ticks

        #remove foods periodically, helps with early training to have more than 1 food as reward is extremely sparse 
        self.remove_schedule = [100_000]
        for i in range(1,n_foods):
            self.remove_schedule.append(self.remove_schedule[-1] + 100_000)
        
        if self.n_foods > 1:
            print(f"Number of starting foods: {n_foods}\nRemove Schedule: {self.remove_schedule}\n")

    
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
                            'head' : ([random.randint(4,self.width-4),
                                      random.randint(4,self.height-4),'up']) #[x coordinate, y coordinate, direction]
        }

        #overlay head onto board array
        self.board[self.snake_head.get('head')[0],
                   self.snake_head.get('head')[1]] = 1

        #snake body, empty on init
        self.snake_body = {
        }
        #number of segments, 0 on init
        self.segments = 0
       
        #spawn initial food 
        self.spawn_food()

        #score and game info
        self.pygame.display.set_caption(f"SCORE: {self.segments}  -  SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")
        self.draw_board()


    #update head location on board
    def update_head_location(self,current_dir,action): 

        #negate illegal moves, can't go backwards           
        if current_dir == 'up' and action    == 'down' or \
           current_dir == 'down' and action  == 'up'   or \
           current_dir == 'right' and action == 'left' or \
           current_dir == 'left' and action  == 'right':
           #if illegal move taken, take last direction 
           action = current_dir

        #conditionals for key presses
        old_hed_pos = self.snake_head.get('head')

        if action == 'up':
            self.new_hed_pos = [old_hed_pos[0],old_hed_pos[1]-1,'up']
        elif action == 'down':
            self.new_hed_pos = [old_hed_pos[0],old_hed_pos[1]+1,'down']
        elif action == 'right':
            self.new_hed_pos = [old_hed_pos[0]+1,old_hed_pos[1],'right']
        elif action == 'left':
            self.new_hed_pos = [old_hed_pos[0]-1,old_hed_pos[1],'left']


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
        delta_reward = 0
        #update head location
        self.update_head_location(current_dir,action)
    
        #head hits wall or self
        if self.board[self.new_hed_pos[0],self.new_hed_pos[1]] == 2 \
            or self.board[self.new_hed_pos[0],self.new_hed_pos[1]] == 4:
                self.done = True
                pygame.time.delay(800)
                self.reduce_food_n()
                self.initialize_game()
                delta_reward = delta_reward - 100                                                   
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
                    col = (65, 105 + random.randint(30,80), 225 + random.randint(0,30))
                elif self.board[r,c] == 3: #food
                    col = (111, 153, 64)
                elif self.board[r,c] == 0:
                    col = ((5,1,74))
                self.pygame.draw.rect(self.surface, col, (r*self.cellsize, c*self.cellsize, 
                                                                self.cellsize-1, self.cellsize-1)) #draw new cell
            else: #wall
                col = (0,139,139)
                self.pygame.draw.rect(self.surface, col,(r*self.cellsize, c*self.cellsize, 
                                                                self.cellsize, self.cellsize)) #draw new cell
        pygame.display.update() #updates display from new .draw in update function


    #spawn food
    def spawn_food(self):
        #spawn food in a loop until not overlapping with any other game entities
        while True:
            not_overlapping = True

            #generate list of tuple-coordinates for food
            self.food = list(tuple((random.randint(1,self.width),
                                random.randint(1,self.height)) for _ in range(self.n_foods)))

            #check if food coords are overlapping
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
            if self.game_ticks > self.remove_schedule[0] and self.n_foods > 1:
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

        self.pygame.display.set_caption(f"SESSION HIGHSCORE: {self.session_highscore}  -  GAMES PLAYED: {self.n_games}")


    #report observation
    def get_observation(self):
        #capture state as np array, convert to tensor for torch
        state = np.transpose(self.board)
        state = state.reshape(1,1,self.height+1,self.width+1)
        state = torch.from_numpy(state)
        
        #normalization of pixel values,  max value is 3
        state = state.float() / 4
        return state.cpu().detach().numpy().squeeze()

        #TODO cleanup :):):)








if __name__ == "__main__":

 # ------------------ Initialize Env ----------------- #
    Environment = Environment_Snake(pygame,
                      width = 16,
                      height = 16,
                      cellsize = 44,
                      fr = 65,
                      n_games = 0,
                      n_foods = 1,
                      game_ticks = 0
                     )
    
    Environment.initialize_board()
    Environment.initialize_game()

    frame = 0

# ------------------ main play loop ------------------ #
    RUNNING = True
    pygame.init()
    
    while RUNNING:     
        
        keypress = False
        for event in pygame.event.get(): 
            #exit window
            if event.type == pygame.QUIT:
                RUNNING = False
                pygame.quit()
                quit()
                
            elif event.type == KEYDOWN:
                pygame.time.delay(Environment.fr)
                #actions here
                Environment.update_environment((pygame.key.name(event.key)),Environment.snake_head.get('head')[2])
                keypress = True
                Environment.draw_board()
                frame+=1
                #pygame.image.save(Environment.surface, f"{frame}.png")
       
        #if there is no user input, update function takes last dir
        if keypress == False:
            pygame.time.delay(Environment.fr)
            Environment.update_environment(Environment.snake_head.get('head')[2],Environment.snake_head.get('head')[2])
            Environment.draw_board()
            frame+=1
            #pygame.image.save(Environment.surface, f"{frame}.png")
    

        














        

        