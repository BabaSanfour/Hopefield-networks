import numpy as np

import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    """
    A Classical Hopfield Network with Hebbian learning.
    
    Attributes:
        num_neurons (int): The number of neurons (N) in the network.
        weights (ndarray): The N x N weight matrix.
    """
    
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """
        Learns the connection weights based on the Hebbian rule: 
        "Neurons that fire together, wire together."
        
        Mathematical rule: w_ij = (1/N) * sum_over_memories(p_i * p_j)
        
        Args:
            patterns (ndarray): Shape (num_patterns, num_neurons). 
                                Values must be {-1, 1}.
        """
        n_patterns, n_neurons = patterns.shape
        
        if n_neurons != self.num_neurons:
            raise ValueError(f"Pattern size {n_neurons} does not match network size {self.num_neurons}")
        
        print(f"Storing {n_patterns} patterns...")
        
        # 1. Vectorized Hebbian Learning
        # Instead of loops, we use matrix multiplication: W = P.T @ P
        W = patterns.T @ patterns
        
        # 2. Remove Self-Connections (Autapses)
        # In physics terms, a spin doesn't interact with itself.
        np.fill_diagonal(W, 0)
        
        # 3. Normalize
        # We divide by N. This scaling is crucial for the math analysis later 
        # (keeping the noise variance finite as N grows).
        self.weights = W / self.num_neurons 

    def energy(self, state):
        """
        Calculates the "Hamiltonian" or Energy of the current state.
        E = -0.5 * sum(w_ij * s_i * s_j)
        
        The network dynamics naturally minimize this value.
        """
        # Vectorized calculation: E = -0.5 * s^T * W * s
        return -0.5 * state @ self.weights @ state

    def update(self, state, synchronous=False):
        """
        Performs one update step of the network dynamics.
        
        Args:
            state (ndarray): Current state of neurons {-1, 1}.
            synchronous (bool): 
                - True: Updates all neurons at once (Little dynamics). Faster but can oscillate.
                - False: Updates neurons one by one in random order (Glauber dynamics). 
                  Guaranteed to converge to a local minimum.
        """
        if synchronous:
            # Calculate input field h_i for all neurons
            activations = self.weights @ state
            
            # Apply sign function. 
            # Note: We handle the edge case where activation is exactly 0.
            new_state = np.sign(activations)
            
            # Inertia Fix: If input is 0, keep the previous state (don't force to +1)
            # np.sign(0) returns 0, so we replace 0s with the old state values.
            new_state[new_state == 0] = state[new_state == 0]
            
            return new_state
        
        else:
            # Asynchronous Update (Glauber Dynamics)
            new_state = state.copy()
            
            # Pick a random order to update neurons
            indices = np.random.permutation(self.num_neurons)
            
            for i in indices:
                # Calculate local field h_i = sum(w_ij * s_j)
                activation = np.dot(self.weights[i], new_state)
                
                # Update rule: s_i = sign(h_i)
                if activation > 0:
                    new_state[i] = 1
                elif activation < 0:
                    new_state[i] = -1
                # Else: activation == 0, do nothing (keep current state)
                
            return new_state

    def predict(self, corrupted_pattern, max_steps=100, synchronous=False):
        """
        Runs the network dynamics until the state stops changing (convergence)
        or we hit max_steps.
        """
        state = corrupted_pattern.copy()
        
        for i in range(max_steps):
            new_state = self.update(state, synchronous=synchronous)
            
            # Check for convergence (stability)
            if np.array_equal(new_state, state):
                return new_state
            
            state = new_state
            
        print("Warning: Network did not fully converge within max_steps.")
        return state

# 1. Setup
N = 100 # Number of neurons (pixels)
hopfield = HopfieldNetwork(num_neurons=N)

# 2. Create Memories (Random patterns for simplicity)
# Using random patterns ensures they are orthogonal-ish, which is best for storage.
num_memories = 3
memories = np.random.choice([-1, 1], size=(num_memories, N))

# 3. Train
hopfield.train(memories)

# 4. Test Retrieval
target_memory = memories[0] # Let's try to recall the first memory

# Create a "corrupted" version (flip 20% of the bits)
noise_level = 0.2
n_flips = int(N * noise_level)
indices_to_flip = np.random.choice(N, n_flips, replace=False)

corrupted_input = target_memory.copy()
corrupted_input[indices_to_flip] *= -1 # Flip the signs

# 5. Run the Network
recovered_state = hopfield.predict(corrupted_input, synchronous=False)

# 6. visual check (First 10 neurons)
print("\n--- Results ---")
print(f"Original:  {target_memory[:10]} ...")
print(f"Corrupted: {corrupted_input[:10]} ...")
print(f"Recovered: {recovered_state[:10]} ...")

# Check if we got it back exactly
if np.array_equal(recovered_state, target_memory):
    print("\nSuccess! Memory perfectly recalled.")
elif np.array_equal(recovered_state, -target_memory):
    print("\nSuccess! Recalled the 'negative' image (physically valid).")
else:
    print("\nFailed to recall. The network got stuck in a spurious local minimum.")