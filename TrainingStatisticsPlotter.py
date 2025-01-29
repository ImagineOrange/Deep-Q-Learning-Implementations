# class imports
import numpy as np
import matplotlib.pyplot as plt

class TrainingStatisticsPlotter:
    '''
    Plot inciteful training statistics for Reinforcement Learning Models
    '''
    
    def __init__(self):
        plt.style.use('dark_background')

    def plot(self, data, x_label, y_label, title, color, marker, x_scale=1000, size=4, overlay=None):
        x = np.arange(0, len(data))
        y = np.array(data)
        plt.figure(figsize=(12, 7))
        plt.scatter(x / x_scale, y, s=size, c=color, marker=marker)
        if overlay is not None:
            y_overlay = np.array(overlay)
            plt.scatter(x / x_scale, y_overlay, s=size, c='r', marker='+')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

    def plot_all(self, 
                 running_rewards, 
                 running_qs_min, 
                 running_qs_max, 
                 foods_eaten, 
                 epsilons, 
                 n_foods, 
                 highscore, 
                 PER_beta):

        # Reward
        self.plot(running_rewards, 'Epochs (1000)', 'Average Reward', 
                  'Sliding-Window: Reward Training Epochs', 'r', '+')

        # Q values (Min and Max overlayed)
        self.plot(running_qs_min, 'Epochs (1000)', 'Q Values', 
                  'Q values over Training Epochs', 'b', '+', overlay=running_qs_max)

        # Foods eaten
        self.plot(foods_eaten, 'Game (every 100)', 'Foods Eaten Upon Death', 
                  'Foods Eaten Per Game', 'b', '+', x_scale=100)

        # Epsilons
        self.plot(epsilons, 'Epochs (1000)', 'Epsilon', 
                  'Epsilon Values', 'm', '+')

        # Number of foods
        self.plot(n_foods, 'Epochs (1000)', '# of Food', 
                  'Number of Food During Training', 'c', '+')

        # Highscore
        self.plot(highscore, 'Epochs (1000)', 'Highscore', 
                  'Highscore over Training', 'c', '+')

        # PER Beta
        self.plot(PER_beta, 'Epochs (1000)', 'PER Beta', 
                  'PER Beta over Training', 'c', '+')
        
        plt.show()

    def sampled_priority_distributions(self, PER_priorities, frame, Buffer):
        # Flatten the list of priorities
        recent_priorities = [priority for sublist in PER_priorities for priority in sublist]

        # Create a figure
        plt.figure(figsize=(12, 7))

        # Create a hexbin plot
        plt.hexbin(recent_priorities, np.arange(len(recent_priorities)), gridsize=200, cmap='inferno', alpha=0.6)

        # Overlay the histogram
        plt.hist(recent_priorities, bins=200, color='purple', alpha=0.6, edgecolor='black')

        # Titles and labels
        plt.title(f'Transition Priority Sampling Distribution at Frame {frame} | Beta:{round(Buffer.beta,3)} | Alpha: {Buffer.alpha} | Max Priority: {Buffer.tree.max_priority}')
        plt.xlabel('Priority')
        plt.ylabel('Frequency')

        # Save the combined figure
        plt.savefig(f'Sampling_distribution_combined_{frame}.png')
        plt.close()

    
    
