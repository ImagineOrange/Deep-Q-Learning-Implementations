# Q-Learning-Paper-Implementations
A collection of deep reinforcement learning algorithms experiments using a custom pygame snake environment 


![](https://github.com/ImagineOrange/Deep-RL-Paper-Implementations/blob/main/snake.gif)


---------------------------------------------------------------------------------------------------------------------------
### human_playable_snake.py
Try out the environment that our RL agents will attempt to learn!

<ins>**Requirements:**</ins> 
_numpy, pygame, random_

<ins>**Description:**</ins> _A homemade snake game built using pygame. The rules are basic, with four directional actions possible,
score increasing for every food block eaten, and the game ending upon the head of the snake hitting a wall or its own body. 
This game will serve as the environment for each deep RL experiment in this repo._ 

<ins>**To run:**</ins> 
_I suggest creating a custom conda environment via command line and then installing requirements using pip :_ 

% conda create --name {env_name} {python==3.9.15}

% conda activate {env_name}

% pip install numpy,pygame,random

% cd {path to downloaded folder containing the code for this repo}

% python human_playable_snake.py

---------------------------------------------------------------------------------------------------------------------------
<ins>**To Train:**</ins> 
Run the _train.py script, I usually see results around 2-4 million frames. Check parameters -- .tar model parameters are saved into the working directory every 500_000 frames. 
Currently, the model is configured to utilize Metal Performance Shaders (MPS) -- GPU accelerated training for mac. 

---------------------------------------------------------------------------------------------------------------------------
<ins>**To watch and evaluate an already-trained model:**</ins> 
download the .tar from the url (google drive) in the _model_url.txt document -- copy the path into the 'agent_training_checkpoint' variable in the main function of the _eval.py script, and then run the script. 








