#environment imports
import numpy as np
import pygame 
import random

#memory replay data structures
from collections import deque

#plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#agent imports
import torch
import torch.nn as nn
import torch.optim

#old numpy warning
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#class imports
from Dueling_DoubleDQN_model import Dueling_DoubleDQN
from Environment_Snake import Environment_Snake

#use CPU or GPU
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"\nDevice used: {device} \n")


'''
    This script evaluates trained models. Models parameters are saved incrementally during training, at the frequency
    'save_every_n_frames' parameter in the training script. Load a model via the 'model_path' variable in the __main__ functiom below. 
    The script automatically measures performance over 'n_evaluations', also in __main__ function. 
    Additionally, the visualize_maps function generates both feature and activation mapping of the convolutional layers in the 
    trained model, which allows for quick debugging and training verification if model performance is sub par. 
'''


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

#Hook to capture the output of a specific layer
activation_maps = {}
feature_maps = {}
sample_frames = []

def hook_fn_activations(module, input, output):
    activation_maps[module] = output

def hook_fn_features(module, input, output):
    feature_maps[module] = output  # Save the raw feature maps (before activation)


def visualize_maps(DQN_, sample_frames, mode="activation"):
    """
    Visualize activation maps or feature maps for the specified Dueling Double DQN model.

    Args:
        DQN_ (nn.Module): The neural network model (Dueling Double DQN).
        sample_frames (list): List of sample input frames (tensors) to visualize.
        mode (str): Mode of visualization - "activation" or "feature".
    """
    assert mode in {"activation", "feature"}, "Mode must be 'activation' or 'feature'."

    # Clear the appropriate map storage
    if mode == "activation":
        activation_maps.clear()
        hook_fn = hook_fn_activations
    else:
        feature_maps.clear()
        hook_fn = hook_fn_features

    # Register hooks for the specified layers
    layers_to_hook = [DQN_.conv_features[0], DQN_.conv_features[2], DQN_.conv_features[4]]  # Indices of Conv2D layers
    for layer in layers_to_hook:
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(hook_fn)

    # Ensure there are frames to process
    if len(sample_frames) == 0:
        print("No frames to visualize.")
        return

    # Convert the first frame to a PyTorch tensor if necessary
    input_tensor = torch.tensor(sample_frames[0], dtype=torch.float32)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension if missing

    # Process and visualize each layer's outputs
    current_input = input_tensor
    for i, layer_idx in enumerate([0, 2, 4]):
        # Forward pass through the layer
        current_input = DQN_.conv_features[layer_idx](current_input)

        # Retrieve the corresponding maps
        if mode == "activation":
            #pull layer-specific learned prelu
            layer_prelu = DQN_.conv_features[layer_idx + 1]
            #pass the kernel feature maps through learned prelu, producing activation maps
            layer_maps = layer_prelu(activation_maps[DQN_.conv_features[layer_idx]]).detach().cpu().numpy()[0]
        else:
            layer_maps = feature_maps[DQN_.conv_features[layer_idx]].detach().cpu().numpy()[0]

        # Limit the number of feature maps to display
        num_features = layer_maps.shape[0]
        max_features = min(32, num_features)  # Limit to 32 feature maps
        cols = 8
        rows = (max_features + cols - 1) // cols  # Calculate rows for subplots

        # Create subplots for visualization
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()

        for j, ax in enumerate(axes):
            if j < max_features:
                ax.imshow(layer_maps[j], cmap='magma')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide unused subplots
        plt.tight_layout()
        
        # Add title indicating the mode and layer
        fig.suptitle(f"Sample {mode.capitalize()} Maps - Layer {i + 1}", fontsize=10)

    # plot inputted sample frames --- dont double plot when function is called twice
    if mode == 'feature':
        # Visualize the raw input frames
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # Create subplots
        axes = axes.flatten()
        for i in range(6):
            axes[i].imshow(input_tensor[0, i].cpu().numpy(), cmap='viridis', interpolation='nearest')
            axes[i].set_title(f'Frame {i+1}')
            axes[i].axis('off')  # Hide axes for clarity
        plt.tight_layout()
        plt.title('INPUT FRAMES FOR FEATURE / ACTIVATION MAPPING')

    print(f"{mode.capitalize()} maps visualized.")


if __name__ == "__main__":

    #init Environment and Agent params
    frame = 0
    games_played = 0
    n_evaluations = 1000
    #animate?
    animate = True
    #load model path for trained model
    model_path = 'MODEL PATH'
    #e.g. : 
    #model_path = '/Users/.../.../PER_1_27/DQN_agent_episodes_94350.pth.tar'

    
    #how many frames to stack for model input
    frame_stack = 6
    state_space = deque([],maxlen=frame_stack)
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

    DQN_ = Dueling_DoubleDQN(input_shape = [frame_stack,Environment.width+1,
                                Environment.height+1],
                                n_actions = 4, epsilon_min=0, epsilon_start=0
                                )
    

    #eval -------------------------------------------------- 

    #load path
    agent_training_checkpoint = torch.load(model_path)
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
                    plotting(Environment.gamescores,Environment)
                    
                    # Visualize activation maps
                    visualize_maps(DQN_, sample_frames, mode="activation")

                    # Visualize feature maps
                    visualize_maps(DQN_, sample_frames, mode="feature")
                    plt.show()
                    exit()

        #sample for feature mapping
        if Environment.segments == 20:
            sample_frames.append(state_tensor)
        
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
    


            


            
    

    
    













        

























