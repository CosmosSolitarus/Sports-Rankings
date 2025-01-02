import numpy as np
import matplotlib.pyplot as plt

def get_steady_state(P):
    eigenvals, eigenvecs = np.linalg.eig(P.T)
    one_index = np.argmin(np.abs(eigenvals - 1))
    steady_state = eigenvecs[:, one_index].real
    return steady_state / np.sum(steady_state)

def create_transition_matrix(q):
    return np.array([
        [1 - 2/9 - (1-q)/3, 1/9, 1/9, (1-q)/3],
        [2/9, 5/9, 1/9, 1/9],
        [2/9, 2/9, 4/9, 1/9],
        [q/3, 2/9, 2/9, 1 - 4/9 - q/3]
    ])

# Create data points
q_values = np.arange(-1, 2.01, 0.01)
steady_states = np.array([get_steady_state(create_transition_matrix(q)) for q in q_values])

# Original steady state values
original_values = [4/11, 7/22, 5/22, 1/11]

# Create the plot
plt.figure(figsize=(12, 8))
states = ['A', 'B', 'C', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot the varying steady states
for i, state in enumerate(states):
    plt.plot(q_values, steady_states[:, i], label=f'State {state}', color=colors[i], linewidth=2)
    
    # Add dotted horizontal line for original value
    plt.axhline(y=original_values[i], color=colors[i], linestyle=':', alpha=0.7)

plt.xlabel('q', fontsize=12)
plt.ylabel('Steady State Probability', fontsize=12)
plt.title('Steady State Probabilities vs. q\nDotted lines show original values', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

plt.show()