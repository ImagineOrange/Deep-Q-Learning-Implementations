import torch
import torch.nn as nn
import numpy as np

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
            for pool_size in [1, 2, 4][:n_spatial_pools]  # Creates pools of different sizes
        ])
        
        # Process each pooled representation separately
        pool_sizes_sum = sum(i * i for i in [1, 2, 4][:n_spatial_pools])
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
        



'''
# Understanding Multi-scale Pooling in Snake

## Conceptual Foundation

Think about how a human plays Snake. When you play, your eyes naturally work at multiple scales simultaneously:
- You look very closely at where the snake's head is
- You maintain awareness of the immediate surrounding area
- You keep track of the overall board state

Our multi-scale pooling architecture mimics this natural visual processing by creating three distinct but complementary views of the game state.

## Technical Implementation

Let's break down how we implement this with our three pooling levels:

```python
self.spatial_pools = nn.ModuleList([
    nn.AdaptiveAvgPool2d((pool_size, pool_size))
    for pool_size in [1, 2, 4]  # Creates pools of different sizes
])
```

### Level 1: 1x1 Pooling (Global Average)
For an 18x18 game board, this level reduces each feature map to a single value. This captures very broad patterns:
- Overall snake length
- General game progression
- Total space utilization

Example using a feature map detecting snake body:
```
Original (18x18):        1x1 Pool:
[0 1 1 0 0 ...]         
[0 0 1 1 0 ...]    →    [0.15]  (single average value)
[0 0 0 1 1 ...]
...
```

### Level 2: 2x2 Pooling (Quadrant Information)
This level divides the board into four quadrants, preserving some spatial relationships:
- Which quadrants contain the snake
- Which quadrant has the food
- Relative emptiness of different board sections

Example:
```
Original (18x18):        2x2 Pool:
[0 1 1 | 0 0 ...]       [0.3  0.1]
[0 0 1 | 1 0 ...]   →   [0.2  0.4]
--------+--------
[0 0 0 | 1 1 ...]
...
```

### Level 3: 4x4 Pooling (Regional Detail)
This provides a more detailed spatial summary:
- Regional snake density
- More precise food location
- Finer-grained empty space detection

Example:
```
Original (18x18):        4x4 Pool:
[0 1 1 | 0 0 | ...]     [0.4  0.1  0.2  0.0]
[0 0 1 | 1 0 | ...]     [0.2  0.3  0.1  0.2]
--------+------+----  →  [0.1  0.4  0.3  0.1]
[0 0 0 | 1 1 | ...]     [0.0  0.2  0.4  0.3]
...
```

## Information Synthesis

The power of this approach comes from how we combine these scales:

```python
# Combine pooled features
pooled_features = []
for pool in self.spatial_pools:
    pooled = pool(x)
    pooled_features.append(pooled.flatten(1))
global_features = torch.cat(pooled_features, dim=1)
```

For each feature map, we now have:
- 1 value from 1x1 pooling (global)
- 4 values from 2x2 pooling (quadrants)
- 16 values from 4x4 pooling (regional)

This gives us 21 values per feature map, each capturing different scales of spatial information. 
With multiple feature maps, we get a rich representation of the game state at multiple scales.

## Practical Benefits for Snake

1. **Path Planning**
   - 4x4 pooling helps identify clear paths to food
   - 2x2 pooling helps avoid trapping the snake in a quadrant
   - 1x1 pooling helps monitor overall space usage

2. **Collision Avoidance**
   - 4x4 pooling detects regional snake density
   - 2x2 pooling identifies safer quadrants
   - Combined information helps plan escape routes

3. **Food Collection Strategy**
   - 4x4 pooling provides relatively precise food location
   - 2x2 pooling helps plan general approach direction
   - 1x1 pooling maintains awareness of overall game state

## Implementation Considerations

1. **Memory Usage**
   The total pooled features per channel:
   - 1x1: 1 value
   - 2x2: 4 values
   - 4x4: 16 values
   Total: 21 values per channel, compared to 324 (18x18) in original

2. **Computational Efficiency**
   - AdaptiveAvgPool2d is highly optimized
   - Parallel processing of different scales
   - Minimal overhead compared to single-scale pooling

Would you like me to elaborate on any particular aspect of the multi-scale pooling or explain how we could modify the scales for different board sizes?






This Improved Global Feature Block is a great enhancement over the original design, especially for Snake, where understanding both local and global spatial relationships is key. Here’s my analysis:

Strengths of the Updated Architecture
✅ Multi-Scale Awareness
The use of 1x1, 2x2, and 4x4 pooling is an excellent idea.
This captures a global view (1x1), quadrant-level trends (2x2), and regional spatial details (4x4).
This approach is biologically inspired, mimicking how humans process game states at different levels.
✅ Feature Grouping for Specialized Processing
Splitting global features into four separate groups and processing them through 1x1 convolutions before merging helps the network learn distinct aspects of the game state.
The PReLU activations allow for adaptive non-linearity, which can improve learning efficiency.
✅ Efficient Use of Global Information
Instead of just a single pooled feature vector, you’re keeping multiple hierarchical views, which prevents loss of important spatial dependencies.
By combining these processed pooled features with local features, the model can better balance fine-grained control and high-level planning.
Potential Areas for Further Improvement
🔹 Potentially High Dimensionality in global_processor
Your global feature vector is flattened as follows:
1x1 pooling → 1 value per channel
2x2 pooling → 4 values per channel
4x4 pooling → 16 values per channel
Total per channel = 21 values
Total flattened size = in_channels * 21
If in_channels = 64, then global_processor receives 1344 input features, which is quite large.
A possible improvement:
Instead of Linear(in_channels * 21, out_channels * 2),
try Linear(in_channels * 21, out_channels), reducing computational cost.
🔹 Alternative Pooling Strategy?
Instead of fixed 1x1, 2x2, 4x4, consider 1x1, 3x3, 6x6 to create a more smoothly spaced hierarchy.
This could improve learning by reducing hard boundaries between regions.
🔹 Feature Grouping: Why 4?
Your feature grouping splits the global features into four different groups.
This is an interesting choice, but does the game state naturally have four major aspects?
If the game state divides well into four roles (e.g., food tracking, self-collision avoidance, general movement, global strategy), then this makes sense.
Otherwise, you could experiment with 3 or 5 groups to see if different partitions improve learning.
Final Verdict
Your modification significantly improves the ability of the network to process multi-scale information. The hierarchical pooling and feature grouping add rich contextual knowledge, which should improve both short-term decision-making and long-term planning.

Would you like help refining the global_processor to reduce computation, or do you prefer to test it as is first? 🚀

'''