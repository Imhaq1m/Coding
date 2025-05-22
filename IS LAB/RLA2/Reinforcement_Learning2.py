import numpy as np
import random
import matplotlib.pyplot as plt

# Environment with a smaller grid and fewer obstacles


class SimpleEnvWithObstacles:
    def __init__(self):
        self.grid_size = 5  # 3 positions: 0, 1, 2
        self.goal_position = 4  # Goal is at position 2
        self.obstacle_position = 2  # Obstacle is at position 1
        self.state = 0  # Agent starts at position 0

    def reset(self):
        self.state = 0  # Reset the agent's position to the start (position 0)
        return self.state

    def step(self, action):
        # If the action is 1 (move right), move right
        if action == 1:
            # Move right, but not past the goal
            self.state = min(self.state + 1, self.grid_size - 1)
        # If the action is 0 (move left), move left
        elif action == 0:
            # Move left, but not past the start
            self.state = max(self.state - 1, 0)

        # Reward and penalty logic
        if self.state == self.goal_position:
            reward = 10  # Reward for reaching the goal
            done = True  # Episode ends when the goal is reached
        elif self.state == self.obstacle_position:
            reward = -5  # Penalty for hitting an obstacle
            done = False  # The episode continues after hitting an obstacle
        else:
            reward = 0  # No reward for other positions
            done = False  # Episode continues unless the goal is reached

        return self.state, reward, done

# Simple Agent without Q-table


class SimpleAgent:
    def __init__(self):
        self.action_space = 2  # Two possible actions: left (0) or right (1)
        # Learning rate (how much to adjust strategy based on experience)
        self.learning_rate = 0.2
        self.exploration_rate = 1.0  # Exploration rate (random actions)
        self.exploration_decay = 0.99  # Decay rate for exploration over time

    def choose_action(self):
        # Choose a random action based on the exploration rate (exploration vs exploitation)
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1])  # Explore: move left (0) or right (1)
        else:
            return 1 if self.state < 5 else 0  # Exploit: always move right if not at the goal

    def update_exploration(self):
        # Decay exploration rate over time
        self.exploration_rate *= self.exploration_decay


# Initialize environment and agent
env = SimpleEnvWithObstacles()
agent = SimpleAgent()

# Training loop
episodes = 50
rewards = []
positions = []  # To track the agent's position during the episode

for episode in range(episodes):
    state = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    episode_positions = []  # Track positions of agent during this episode

    while not done:
        agent.state = state  # Set the agent's current state
        action = agent.choose_action()  # Choose an action (random or greedy)
        # Take the action and observe the result
        next_state, reward, done = env.step(action)
        total_reward += reward  # Update total reward for this episode
        state = next_state  # Update the state

        episode_positions.append(state)  # Record the agent's position

    rewards.append(total_reward)  # Record the total reward for this episode
    # Store the positions for visualization
    positions.append(episode_positions)
    agent.update_exploration()  # Update the exploration rate (decay over time)

# Plot the rewards (they will vary based on the agent's exploration and learning)
plt.plot(rewards)
plt.title("Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("Rewards_Over_Time.png")
# plt.show()

# Plot the agent's movement in the grid
plt.figure(figsize=(10, 2))
# Plot the first 10 episodes' movements
for episode_positions in positions[:10]:
    plt.plot(episode_positions, np.ones(len(episode_positions)),
             marker='o', linestyle='-', markersize=6)

plt.title("Agent's Movement Over Episodes")
plt.xlabel("Grid Position")
plt.yticks([])  # Hide y-axis because it’s just for visualization of positions
plt.grid(True)
plt.savefig("Agent's Movement Over Episodes.png")
# plt.show()

# Test the learned policy (no exploration)
state = env.reset()
done = False
test_rewards = []
test_positions = []

print("\nTesting the learned policy:")

while not done:
    action = 1 if state < 5 else 0  # Exploit: always move right if not at the goal
    next_state, reward, done = env.step(action)
    test_rewards.append(reward)
    test_positions.append(next_state)
    state = next_state

print(f"Total reward in test episode: {sum(test_rewards)}")

# Plot the test phase agent movement
plt.figure(figsize=(10, 2))
plt.plot(test_positions, np.ones(len(test_positions)),
         marker='o', linestyle='-', color='r', markersize=6)
plt.title("Agent's Movement During Test Phase")
plt.xlabel("Grid Position")
plt.yticks([])  # Hide y-axis
plt.grid(True)
plt.savefig("Agent's_Movement_During_Test_Phase")
# plt.show()
