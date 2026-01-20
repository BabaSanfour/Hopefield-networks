from hopfield import HopfieldNetwork, ModernHopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def get_vacation_patterns():
    # Legend: -1 = White (Background), 1 = Black (Ink)
    
    # Memory 1: Penguin Trip 
    # (A little penguin standing)
    penguin = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1, -1, -1, -1, -1], # Head
        [-1, -1, -1,  1, -1,  1,  1, -1, -1, -1], # Eye/Beak
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1], # Neck
        [-1,  1,  1, -1, -1, -1,  1,  1,  1, -1], # Wings/Belly
        [-1,  1,  1, -1, -1, -1,  1,  1,  1, -1],
        [-1,  1,  1, -1, -1, -1,  1,  1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1], # Body
        [-1, -1,  1, -1, -1, -1,  1, -1, -1, -1], # Feet
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]).flatten()

    # Memory 2: Yoga Session 
    # (Stick figure in Lotus position/Tree pose)
    yoga = np.array([
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1], # Head
        [-1, -1,  1, -1,  1,  1, -1,  1, -1, -1], # Arms up
        [-1, -1, -1,  1,  1,  1,  1, -1, -1, -1],
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1], # Torso
        [-1, -1, -1, -1,  1,  1, -1, -1, -1, -1],
        [-1, -1, -1,  1, -1, -1,  1, -1, -1, -1], # Hips
        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1], # Knees
        [-1, -1,  1,  1, -1, -1,  1,  1, -1, -1], # Legs crossed
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]).flatten()

    # Memory 3: Beach 
    # (Sun in corner, waves at bottom)
    beach = np.array([
        [-1, -1, -1, -1, -1, -1, -1,  1,  1,  1], # Sun
        [-1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1,  1,  1,  1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # Sky
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1], # Wave crest
        [ 1,  1,  1, -1, -1,  1,  1,  1, -1, -1], # Waves
        [-1, -1,  1,  1,  1,  1, -1,  1,  1,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1], # Deep water
        [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    ]).flatten()
    
    return np.array([penguin, yoga, beach]), ["Penguin Trip", "Yoga Session", "Beach"]

def run_vacation_demo():
    N = 100 # 10x10
    
    # 1. Load your memories
    patterns, labels = get_vacation_patterns()
    
    # 2. Initialize and Train
    hopfield = HopfieldNetwork(N)
    hopfield.train(patterns)
    
    # 3. Create a Test Case
    # Let's try to recall the "Yoga Session", but the memory is faded (noisy)
    target_idx = 1
    original = patterns[target_idx]
    
    # Add significant noise (30% of pixels flipped)
    noise_level = 0.30
    n_flips = int(N * noise_level)
    indices = np.random.choice(N, n_flips, replace=False)
    
    corrupted = original.copy()
    corrupted[indices] *= -1 
    
    # 4. Attempt Recovery
    recovered = hopfield.predict(corrupted)
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    def plot_grid(ax, data, title, color='Greys'):
        ax.imshow(data.reshape(10, 10), cmap=color, vmin=-1, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        # Add a nice border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

    # Plot Original
    plot_grid(axes[0], original, f"Original Memory\n({labels[target_idx]})", color='Blues')
    
    # Plot Corrupted
    plot_grid(axes[1], corrupted, f"Corrupted Input\n(Faded Memory: {int(noise_level*100)}% Noise)", color='Greys')
    
    # Plot Recovered
    # Check if it matches ANY of the stored patterns to see if we hallucinated a yoga class instead
    is_match = np.array_equal(recovered, original)
    color = 'Greens' if is_match else 'Reds'
    result_text = "Succesfully Recalled!" if is_match else "Recall Failed / Spurious State"
    
    plot_grid(axes[2], recovered, f"Network Output\n{result_text}", color=color)
    
    plt.suptitle("Imbizo Memories", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("Computing Energy Landscape (PCA Projection)...")

    # A. Use PCA to find the 2D plane spanning our 3 memories
    pca = PCA(n_components=2)
    pca.fit(patterns)
    
    # Get the coordinates of our memories in this 2D space
    memories_2d = pca.transform(patterns)
    
    # B. Create a meshgrid covering this area
    # We add some margin to see the "walls" around the memories
    x_min, x_max = memories_2d[:, 0].min() - 5, memories_2d[:, 0].max() + 5
    y_min, y_max = memories_2d[:, 1].min() - 5, memories_2d[:, 1].max() + 5
    
    resolution = 50
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # C. Calculate Energy for every point on the grid
    Z_energy = np.zeros_like(xx)
    
    for i in range(resolution):
        for j in range(resolution):
            # 1. Map 2D point -> 100D space
            point_2d = np.array([[xx[i,j], yy[i,j]]])
            point_100d = pca.inverse_transform(point_2d)[0]
            
            # 2. Normalize magnitude
            norm = np.linalg.norm(point_100d)
            if norm > 0:
                point_100d = point_100d / norm * np.sqrt(N)
            
            # 3. Compute Energy
            Z_energy[i,j] = hopfield.energy(point_100d)

    # D. Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the landscape surface
    surf = ax.plot_surface(xx, yy, Z_energy, cmap='viridis_r', alpha=0.8, edgecolor='none')
    
    # Plot the actual memories as red dots
    # We compute their energy specifically to place them correctly
    memory_energies = [hopfield.energy(p) for p in patterns]
    ax.scatter(memories_2d[:,0], memories_2d[:,1], memory_energies, 
               color='red', s=100, label='Memories', depthshade=False)

    # Add labels for the memories
    for k, label in enumerate(labels):
        ax.text(memories_2d[k,0], memories_2d[k,1], memory_energies[k], 
                f"  {label}", color='black', fontsize=10, fontweight='bold')

    # Styling
    ax.set_title("Energy Landscape of Imbizo Memories", fontsize=15)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Energy')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_phase_transition_experiment():
    # Parameters
    N = 100               # Number of neurons
    TRIALS = 20           # Trials per grid point (higher = smoother map)
    
    # Grid Ranges
    m_range = np.arange(1, 26, 2) 
    
    # Noise: Percentage of bits flipped (0% to 50%)
    noise_range = np.linspace(0, 0.5, 20) 
    
    # Results Matrix: Shape (len(noise), len(M))
    phase_matrix = np.zeros((len(noise_range), len(m_range)))

    print(f"Running Phase Transition Simulation (N={N})...")
    print(f"Grid size: {len(m_range)} (Memories) x {len(noise_range)} (Noise levels)")

    # --- Simulation Loop ---
    for i, noise_level in enumerate(noise_range):
        for j, M in enumerate(m_range):
            success_count = 0
            
            for _ in range(TRIALS):
                # 1. Generate M random patterns
                patterns = np.random.choice([-1, 1], size=(M, N))
                
                # 2. Train Network
                hn = HopfieldNetwork(N)
                hn.train(patterns)
                
                # 3. Pick a target and corrupt it
                target = patterns[0]
                
                # Create noise mask
                n_flips = int(N * noise_level)
                indices = np.random.choice(N, n_flips, replace=False)
                corrupted = target.copy()
                corrupted[indices] *= -1
                
                # 4. Attempt Retrieval
                recovered = hn.predict(corrupted)
                
                # 5. Check Success
                # We check for exact match OR inverted match (physically valid)
                if np.array_equal(recovered, target) or np.array_equal(recovered, -target):
                    success_count += 1
            
            # Calculate probability
            phase_matrix[i, j] = success_count / TRIALS
        
        # Simple progress bar
        print(f"Completed Noise Level {noise_level:.2%}")

    # --- 3. Visualization ---
    plt.figure(figsize=(10, 8))
    
    # We use extent to map the array indices back to physical units (M and Noise)
    # Origin='lower' puts (0,0) at bottom left
    plt.imshow(phase_matrix, 
               extent=[m_range.min(), m_range.max(), noise_range.min(), noise_range.max()],
               origin='lower', 
               aspect='auto', 
               cmap='RdYlBu_r', # Red=Bad, Blue=Good (or reversed depending on preference)
               vmin=0, vmax=1)
    
    cbar = plt.colorbar()
    cbar.set_label('Retrieval Probability', rotation=270, labelpad=15)

    # Add the Theoretical Limit Line (Capacity ~ 0.14N)
    plt.axvline(x=0.14*N, color='black', linestyle='--', linewidth=2, label='Theoretical Capacity (~0.14N)')

    plt.title(f'Hopfield Network Phase Diagram (N={N})', fontsize=14)
    plt.xlabel('Number of Stored Memories (M)', fontsize=12)
    plt.ylabel('Input Noise Level (Fraction of Bits Flipped)', fontsize=12)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def run_modern_phase_transition():
    # Parameters
    N = 100               # Number of neurons
    TRIALS = 10           # Trials per grid point
    BETA = 5.0            # High beta ensures sharp attention
    
    # --- GRID RANGES (The Big Difference) ---
    # Standard Limit was ~14 memories.
    # Modern Limit is exponential. We will test up to 3xN (300 memories).
    m_range = np.arange(10, 300, 15) 
    
    # Noise: 0% to 50%
    noise_range = np.linspace(0, 0.5, 20) 
    
    # Results Matrix
    phase_matrix = np.zeros((len(noise_range), len(m_range)))

    print(f"Running MODERN Phase Transition (N={N}, Beta={BETA})...")
    print(f"Testing up to {m_range.max()} memories (3x Neurons!)")

    # --- Simulation Loop ---
    for i, noise_level in enumerate(noise_range):
        for j, M in enumerate(m_range):
            success_count = 0
            
            for _ in range(TRIALS):
                # 1. Generate M random patterns
                patterns = np.random.choice([-1, 1], size=(M, N))
                
                # 2. Train Modern Network
                mhn = ModernHopfieldNetwork(N, beta=BETA)
                mhn.train(patterns)
                
                # 3. Corrupt Target
                target = patterns[0]
                n_flips = int(N * noise_level)
                indices = np.random.choice(N, n_flips, replace=False)
                corrupted = target.copy()
                corrupted[indices] *= -1
                
                # 4. Retrieve
                recovered = mhn.predict(corrupted)
                
                # 5. Check Success
                if np.array_equal(recovered, target) or np.array_equal(recovered, -target):
                    success_count += 1
            
            phase_matrix[i, j] = success_count / TRIALS
        
        print(f"Completed Noise Level {noise_level:.2%}")

    # --- 3. Visualization ---
    plt.figure(figsize=(12, 8))
    
    # Plot Heatmap
    plt.imshow(phase_matrix, 
               extent=[m_range.min(), m_range.max(), noise_range.min(), noise_range.max()],
               origin='lower', 
               aspect='auto', 
               cmap='RdYlBu_r', 
               vmin=0, vmax=1)
    
    cbar = plt.colorbar()
    cbar.set_label('Retrieval Probability', rotation=270, labelpad=15)

    # Reference Lines
    plt.axvline(x=0.14*N, color='white', linestyle='--', linewidth=2, label='Old Standard Limit (0.14N)')
    plt.axvline(x=N, color='red', linestyle=':', linewidth=2, label='N Neurons (100)')
    
    plt.title(f'Modern Hopfield Network Phase Diagram (N={N})', fontsize=14)
    plt.xlabel('Number of Stored Memories (M)', fontsize=12)
    plt.ylabel('Input Noise Level', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9, facecolor='black', labelcolor='white')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_vacation_demo()
    run_phase_transition_experiment()
    run_modern_phase_transition()
