
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_inputs, n_action, n_hidden_layers=2, hidden_dim=32):
        super(MLP, self).__init__()
        M = n_inputs
        self.layers = []

        # Create hidden layers
        for _ in range(n_hidden_layers):
            layer = nn.Linear(M, hidden_dim)  # A linear layer, with input size M and output size hidden_dim
            M = hidden_dim  # Set the hidden_dim as the input size for the next layer
            self.layers.append(layer)
            self.layers.append(nn.ReLU())  # A ReLU activation function after each layer

        # Final layer, output size = action_space_dim
        self.layers.append(nn.Linear(M, n_action))

        # Combine all layers into a sequential container
        self.layers = nn.Sequential(*self.layers)

    # Forward pass
    def forward(self, X):
        return self.layers(X)

    # Save and load weights as needed
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


def predict(model, np_states):
    # Predict the Q-function associated with each action given the current state
    with torch.no_grad():
        inputs = torch.from_numpy(np_states.astype(np.float32))
        output = model(inputs)
        return output.numpy()


def train_one_step(model, criterion, optimizer, inputs, targets):
    # Type conversion
    inputs = torch.from_numpy(inputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)

    # Compute the loss term
    loss = criterion(outputs, targets)

    # Backward and optimize
    loss.backward()
    optimizer.step()
    return loss.item()


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        # Buffer initiation, assign space to store current and next observations, actions, rewards, and done flag
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # Store a new experience in the buffer
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # Update the pointer to the next location
        self.size = min(self.size + 1, self.max_size)  # Update the current buffer size

    def sample_batch(self, batch_size=32):
        # Sample a batch of experiences from the buffer
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


class DQNAgent:
    def __init__(self, state_size, action_size, model):
        self.state_size = state_size  # Size of the state space, derived from the dataset fed into MarketEnv
        self.action_size = action_size  # Size of the action space, derived from the number of trading assets
        self.memory = ReplayBuffer(state_size, action_size, size=500)  # Initiate the Replay Buffer
        self.gamma = 0.95  # Discount factor when updating the Q-function
        self.epsilon = 1.0  # Epsilon-greedy strategy set up
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # The decay
        self.model = model  # The neural network model
        self.target_model = model  # Initialize a target model
        self.target_model.load_state_dict(model.state_dict())  # Set the same weights as the training model
        self.criterion = nn.MSELoss()  # Loss function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimizer

    def update_replay_memory(self, state, action, reward, next_state, done):
        # Store a state in the replay buffer
        self.memory.store(state, action, reward, next_state, done)

    def act(self, state):
        # Explore vs Exploit, applying epsilon-greedy strategy
        if np.random.rand() <= self.epsilon:
            # Explore, a random action is chosen
            return np.random.choice(self.action_size)

        # Exploit, predict the Q-functions given the current state
        act_values = predict(self.model, state)

        # Return the action index that gives the highest Q-value
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        if self.memory.size < batch_size:
            return

        # Sample a batch of experience from the replay memory for training
        minibatch = self.memory.sample_batch(batch_size)
        states = minibatch['s']
        actions = minibatch['a']
        rewards = minibatch['r']
        next_states = minibatch['s2']
        done = minibatch['d']

        # Predict the target Q-values Q(s',a) using the sample batch and target model
        target = rewards + (1 - done) * self.gamma * np.amax(predict(self.target_model, next_states), axis=1)
        target_full = predict(self.model, states)
        target_full[np.arange(batch_size), actions] = target

        # Run one training step using the training model
        train_one_step(self.model, self.criterion, self.optimizer, states, target_full)

        # Update the exploration decay after each training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # Update the target model weights to match the model weights
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        # Load a trained model weight from a specified file
        self.model.load_weights(name)

    def save(self, name):
        # Save model weights for future use
        self.model.save_weights(name)
