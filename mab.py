import numpy as np
import matplotlib.pyplot as plt
 
class TenArmedBanditEpsilonGreedy:
    def __init__(self, num_arms=10, mean_rewards=None, std_dev=1.0, epsilon=0.1):
        self.num_arms = num_arms
        self.std_dev = std_dev
        self.epsilon = epsilon
        
        if mean_rewards is None:
            self.mean_rewards = np.random.normal(0, 1, self.num_arms)
        else:
            if len(mean_rewards) != self.num_arms:
                raise ValueError("mean_rewards should have {} elements.".format(self.num_arms))
            self.mean_rewards = np.array(mean_rewards)
        
        self.action_values = np.zeros(self.num_arms)
        self.action_counts = np.zeros(self.num_arms)
        
        self.time_step = 0
    
    def reset(self):
        self.time_step = 0
        self.action_values = np.zeros(self.num_arms)
        self.action_counts = np.zeros(self.num_arms)
    
    def get_true_rewards(self):
        return self.mean_rewards
    
    def get_action_space(self):
        return range(self.num_arms)
    
    def select_action(self):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_arms)
        else:
            action = np.argmax(self.action_values)
        return action
    
    def step(self):
        action = self.select_action()
        reward = np.random.normal(self.mean_rewards[action], self.std_dev)
        
        self.time_step += 1
        
        self.action_counts[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.action_counts[action]
        
        return action, reward
 
# Define the parameters
num_arms = 10
num_steps = 1000  # Number of time steps
num_runs = 2000    # Number of independent runs
epsilons = [0.1, 0.01, 0.0]  # Different epsilon values to compare
 
# Initialize arrays to store the average rewards and optimal action percentages for each epsilon value
average_rewards = np.zeros((len(epsilons), num_steps))
optimal_action_percentages = np.zeros((len(epsilons), num_steps))
 
# Run experiments for each epsilon value
for i, epsilon in enumerate(epsilons):
    avg_reward_per_step = np.zeros(num_steps)
    optimal_action_count = np.zeros(num_steps)
    
    for run in range(num_runs):
        bandit = TenArmedBanditEpsilonGreedy(num_arms=num_arms, epsilon=epsilon)
        rewards = np.zeros(num_steps)
        
        for step in range(num_steps):
            action, reward = bandit.step()
            rewards[step] = reward
            if action == np.argmax(bandit.get_true_rewards()):
                optimal_action_count[step] += 1
            
        avg_reward_per_step += (rewards - avg_reward_per_step) / (run + 1)
    
    average_rewards[i, :] = avg_reward_per_step
    optimal_action_percentages[i, :] = (optimal_action_count / num_runs) * 100  # Calculate the percentage
 
# Plot the results
plt.figure(figsize=(12, 6))
 
# Plot Average Reward vs. Number of Steps
plt.subplot(1, 2, 1)
for i, epsilon in enumerate(epsilons):
    plt.plot(range(1, num_steps + 1), average_rewards[i, :], label=f"Epsilon = {epsilon}")
plt.xlabel("Time Step")
plt.ylabel("Average Reward")
plt.yticks(np.arange(0, 2.5, 0.5))
plt.legend()
plt.title("Epsilon-Greedy Action Selection (Average Reward)")
plt.grid(True)
 
# Plot Optimal Action Percentage vs. Number of Steps
plt.subplot(1, 2, 2)
for i, epsilon in enumerate(epsilons):
    plt.plot(range(1, num_steps + 1), optimal_action_percentages[i, :], label=f"Epsilon = {epsilon}")
plt.xlabel("Time Step")
plt.ylabel("Optimal Action %")
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.title("Epsilon-Greedy Action Selection (Optimal Action %)")
plt.grid(True)
 
plt.tight_layout()
plt.show()
