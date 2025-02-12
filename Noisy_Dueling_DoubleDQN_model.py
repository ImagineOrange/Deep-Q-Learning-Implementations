import torch
import torch.nn as nn
import numpy as np

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer with Independent or Factorized Gaussian Noise.
    """
    def __init__(self, in_features, out_features, factorized=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(0.017)  # Default sigma initialization from Fortunato et al. 2018
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(0.017)

    def forward(self, x):
        device = self.weight_mu.device  # Ensure all random tensors are on the same device

        if self.factorized:
            # Factorized Gaussian Noise
            noise_input = self._noise(self.in_features, device)
            noise_output = self._noise(self.out_features, device)
            weight_noise = torch.outer(noise_output, noise_input)
            bias_noise = noise_output
        else:
            # Independent Gaussian Noise
            weight_noise = torch.randn_like(self.weight_sigma, device=device)
            bias_noise = torch.randn_like(self.bias_sigma, device=device)

        weight = self.weight_mu + self.weight_sigma * weight_noise
        bias = self.bias_mu + self.bias_sigma * bias_noise

        return torch.nn.functional.linear(x, weight, bias)

    def get_noise_level(self):
        return self.weight_sigma.abs().mean().item(), self.bias_sigma.abs().mean().item()

    @staticmethod
    def _noise(size, device):
        noise = torch.randn(size, device=device)  # Ensure the noise is on the correct device
        return noise.sign() * torch.sqrt(torch.abs(noise))  # Factorized noise


class ImprovedGlobalFeatureBlock(nn.Module):
    """
    Improved Global Feature Block that preserves multiple types of global features
    through spatial pooling pyramids and feature grouping.
    
    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output feature channels
        n_spatial_pools: Number of different spatial pooling levels
    """
    def __init__(self, in_channels, out_channels, n_spatial_pools=3):
        super().__init__()
        
        self.n_spatial_pools = n_spatial_pools
        
        # Local pathway preserves full spatial resolution
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # Multiple pooling levels to preserve different scales of information
        self.spatial_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((pool_size, pool_size))
            for pool_size in [1, 3, 6][:n_spatial_pools]  # Creates pools of different sizes
        ])
        
        # Process each pooled representation separately
        pool_sizes_sum = sum(i * i for i in [1, 3, 6][:n_spatial_pools])
        self.global_processor = nn.Sequential(
            nn.Linear(in_channels * pool_sizes_sum, out_channels * 2),
            nn.PReLU(),
            nn.Linear(out_channels * 2, out_channels),
            nn.PReLU()
        )
        
        # Feature grouping for different aspects of game state
        self.feature_groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.PReLU()
            ) for _ in range(4)  # 4 different feature groups
        ])
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Local pathway
        local_features = self.local_conv(x)
        
        # Multi-scale pooling pathway
        pooled_features = []
        for pool in self.spatial_pools:
            pooled = pool(x.cpu())  # Move tensor to CPU before pooling
            pooled_features.append(pooled.to(x.device).flatten(1))  # Move it back to MPS
        
        # Combine pooled features
        global_features = torch.cat(pooled_features, dim=1)
        global_features = self.global_processor(global_features)
        
        # Split into feature groups and process separately
        feature_groups = []
        for group in self.feature_groups:
            # Reshape global features to 2D and process
            group_input = global_features.view(batch_size, -1, 1, 1)
            processed_group = group(group_input)
            # Expand to original spatial dimensions
            expanded = processed_group.expand(-1, -1, local_features.size(2), local_features.size(3))
            feature_groups.append(expanded)
            
        # Combine all feature groups
        global_context = torch.cat(feature_groups, dim=1)
        
        # Final combination of local and multi-scale global features
        return local_features + global_context

    def get_feature_maps(self, x):
        """Helper method to visualize different feature groups"""
        with torch.no_grad():
            local = self.local_conv(x)
            pooled_maps = [pool(x) for pool in self.spatial_pools]
            return local, pooled_maps
        


class Noisy_Dueling_DoubleDQN(nn.Module):
    """
    Dueling Double DQN architecture using Noisy Networks for exploration.
    Includes global feature processing for enhanced spatial awareness.
    Args: 
        input_shape   : [frames stacked x game width x game height] 
        n_actions     : number of possible actions to generate Q-values outputs
        factorized_noise : whether to use factorized noise in NoisyLinear layers
    """
    def __init__(self, input_shape, n_actions, factorized_noise=True):
        super().__init__()

        self.factorized_noise = factorized_noise

        # Initial feature extraction while preserving spatial information
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        
        # Global feature processing block
        self.global_block = ImprovedGlobalFeatureBlock(64, 128)
        
        # Optional downsampling for memory efficiency
        self.downsample = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.PReLU()
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape), 256),
            nn.PReLU(),
            NoisyLinear(256, n_actions, factorized=self.factorized_noise),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.get_conv_out(input_shape), 256),
            nn.PReLU(),
            NoisyLinear(256, 1, factorized=self.factorized_noise),
        )

    def get_conv_out(self, input_shape):
        """
        Calculate the size of the flattened features after convolution and global processing.
        Args:
            input_shape : [frames stacked x game width x game height]
        Returns:
            int : Size of flattened feature vector
        """
        device = next(self.parameters()).device
        o = self.conv_features(torch.zeros(1, *input_shape, device=device))
        o = self.global_block(o)
        o = self.downsample(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # Extract initial features
        conv_out = self.conv_features(x)
        
        # Process global and local features
        features = self.global_block(conv_out)
        
        # Downsample and flatten
        features = self.downsample(features)
        features = features.view(x.size()[0], -1)
        
        # Value and advantages
        value = self.value_stream(features)
        actions_advantages = self.advantage_stream(features)
        average_advantages = torch.mean(actions_advantages, dim=1, keepdim=True)

        Q = value + (actions_advantages - average_advantages)
        return Q

    def random_action(self, observation):
        """
        For evaluation: Generates a random action with a small probability.
        Args:
            observation : Current game state observation
        Returns:
            torch.Tensor : Selected action
        """
        p_random = 5000
        random_sample = np.random.randint(1, p_random + 1)
        if p_random == random_sample:
            action = np.random.uniform(-1, 0, 4)
            return torch.tensor(action)
        else:
            return self.forward(observation)