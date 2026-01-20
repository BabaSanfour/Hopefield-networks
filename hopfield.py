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


class ModernHopfieldNetwork:
    """
    Dense Associative Memory (Modern Hopfield Network).
    """
    def __init__(self, num_neurons, beta=1.0):
        self.num_neurons = num_neurons
        self.patterns = None
        self.beta = beta # Inverse temperature (1/T). High beta = Argmax behavior.

    def train(self, patterns):
        """
        In Modern Hopfield Nets, 'training' is just storing the matrix P.
        We do NOT compute a weight matrix W (size N x N).
        Instead, we keep the raw patterns (size M x N).
        """
        self.patterns = patterns
        self.num_patterns = patterns.shape[0]

    def update(self, state):
        """
        Update rule: x_new = sign( P.T @ softmax(beta * P @ x) )
        This is Attention(Q, K, V) where Q=state, K=V=patterns.
        """
        # 1. Similarity (Dot Product)
        # Check how well the state matches EACH memory.
        # Shape: (M, N) @ (N,) -> (M,)
        overlap = self.patterns @ state
        
        # 2. Attention Weights (Softmax)
        # We apply the exponential function to create a "winner-take-all" effect.
        logits = self.beta * overlap
        
        # Numerically stable softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)
        
        # 3. Weighted Reconstruction
        # Construct the new state as a weighted average of ALL memories.
        # If beta is high, 'probs' will be [0, 0, 1, 0], effectively copying the best match.
        # Shape: (N, M) @ (M,) -> (N,)
        new_continuous = self.patterns.T @ probs
        
        # 4. Binarize (for Ising model compatibility)
        # Note: Modern Hopfield nets can work with continuous states, 
        # but for this homework comparing to Classical, we enforce binary states.
        # Handle the 0 case (inertia)
        new_binary = np.sign(new_continuous)
        new_binary[new_binary == 0] = state[new_binary == 0] 
        
        return new_binary

    def predict(self, corrupted_pattern, max_steps=5):
        # Modern networks converge EXTREMELY fast (often 1 step).
        state = corrupted_pattern.copy()
        for i in range(max_steps):
            new_state = self.update(state)
            if np.all(new_state == state):
                return new_state
            state = new_state
        return state
