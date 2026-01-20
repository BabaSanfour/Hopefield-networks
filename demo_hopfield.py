from hopfield import HopfieldNetwork
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    run_vacation_demo()