import numpy as np
import json
from collections import deque
import time # revise
from gridworld import GridWorld
import random
# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)


    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        returns = {state: [] for state in range(self.state_space)}
        while self.episode_counter < self.max_episode:
            episode = []
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                episode.append((current_state, reward)) 
                current_state = next_state  
            
            G = 0 
            visited_states = set()  
            G_t = np.zeros(len(episode))
            for t in reversed(range(len(episode))):
                state, reward = episode[t]
                G = reward + self.discount_factor * G 
                G_t[t] = G
            for t in range(len(episode)):
                state, reward = episode[t]
                if state not in visited_states:
                    visited_states.add(state)
                    returns[state].append(G_t[t])  
                    self.values[state] = np.mean(returns[state]) 
            


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                td_target = reward + self.discount_factor * self.values[next_state] 
                if done:
                    td_target = reward
                td_error = td_target - self.values[current_state]  
                self.values[current_state] += self.lr * td_error  
                current_state = next_state


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            done = False
            # Initialize the state, reward, and next_state buffers for n-step TD updates
            state_buffer = []  # Buffer to store states
            reward_buffer = []  # Buffer to store rewards
            
            # Interact with the environment
            while not done:
                # Collect data: current state, reward, and next state
                next_state, reward, done = self.collect_data()
                
                # Append the current state and reward to the buffers
                state_buffer.append(current_state)
                reward_buffer.append(reward)
                
                # If we have more than n states in the buffer, we can start updating the value function
                if len(state_buffer) >= self.n:
                    G = 0
                    for i in range(self.n):
                        G += (self.discount_factor ** i) * reward_buffer[i]
                    if not done: 
                        G += (self.discount_factor ** self.n) * self.values[next_state]  
                    td_error = G - self.values[state_buffer[0]]
                    self.values[state_buffer[0]] += self.lr * td_error
                    state_buffer.pop(0)
                    reward_buffer.pop(0)
                current_state = next_state

            while len(state_buffer) > 0:
                G = 0
                for i in range(len(reward_buffer)):
                    G += (self.discount_factor ** i) * reward_buffer[i]
                td_error = G - self.values[state_buffer[0]]
                self.values[state_buffer[0]] += self.lr * td_error
                state_buffer.pop(0)
                reward_buffer.pop(0)





# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)

        G = 0 
        for t in reversed(range(len(action_trace))):
            state = state_trace[t]
            action = action_trace[t]
            reward = reward_trace[t]
            G = reward + self.discount_factor * G  
            self.q_values[state, action] += self.lr * (G - self.q_values[state, action]) 
        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        policy_indices = self.get_policy_index()
    
        # Iterate over all states and update the policy
        for state in range(self.state_space):
            action_probabilities = np.ones(self.action_space) * (self.epsilon / self.action_space)
            best_action = policy_indices[state]
            action_probabilities[best_action] += (1.0 - self.epsilon)
            self.policy[state] = action_probabilities

        # # Update the policy for the current state
        # 
        # for state in range(self.state_space):
        #     action_probabilities = np.ones(self.action_space) * (self.epsilon / self.action_space)
        #     best_action = np.argmax(self.q_values[state])
        #     action_probabilities[best_action] += (1.0 - self.epsilon)
        #     self.policy[state] = action_probabilities


    def run(self, max_episode=1000) -> None: ### im main function , it is 512000 iterations... 
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        start_time = time.time()

        iter_episode = 0
        current_state = self.grid_world.reset()
        state_trace   = [current_state]
        action_trace  = []
        reward_trace  = []
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # print('current state: ',current_state)
            state_trace   = [current_state]
            action_trace  = []
            reward_trace  = []
            # Generate one episode
            done = False
            while not done:
                
                action_probs = self.policy[current_state]
                action = np.random.choice(self.action_space, p=action_probs)
                next_state, reward, done = self.grid_world.step(action)
                current_state = next_state
                state_trace.append(current_state)
                action_trace.append(action)
                reward_trace.append(reward)
                
            # print(iter_episode)
            # print('action_trace',len(action_trace))
            # Evaluate the policy by updating Q-values using Every-Visit Monte Carlo
            self.policy_evaluation(state_trace, action_trace, reward_trace)
            # Improve the policy using epsilon-greedy improvement
            self.policy_improvement()
            iter_episode += 1
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time  # Calculate total running time
        print(f"Total running time: {total_time:.2f} seconds")

class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        # Evaluate Q value after one step and improve the policy using TD(0)
        target = 0
        if is_done:
            target = r  # If the episode is done, there is no future reward
        else:
            target = r + self.discount_factor * self.q_values[s2, a2]
        
        td_error = target - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error

        # Improve the policy at state s using epsilon-greedy

        policy_indices = self.get_policy_index()    
        best_action = policy_indices[s]
        action_probabilities = np.ones(self.action_space) * (self.epsilon / self.action_space)
        action_probabilities[best_action] += (1.0 - self.epsilon)
        self.policy[s] = action_probabilities

        # raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        start_time = time.time()
        iter_episode = 0
        current_state = self.grid_world.reset()
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            # print('current_state:', current_state)
            prev_s = None
            prev_a = None
            prev_r = None
            action_probs = self.policy[current_state]
            action = np.random.choice(self.action_space, p=action_probs)
            done = False
            while not done:
                # Take action, observe reward and next state
                next_state, reward, done = self.grid_world.step(action)
                # if done:
                #     print('prev of done state:', current_state) # must be 21
                #     print('after final state', reward)
                #     print('done state:', next_state)
                prev_a = action
                prev_r = reward
                prev_s = current_state
                # Choose next action using the current policy (epsilon-greedy)
                if not done:
                    next_action_probs = self.policy[next_state]
                    next_action = np.random.choice(self.action_space, p=next_action_probs)
                else:
                    next_action = None

                current_state = next_state
                action = next_action

                # Update Q-values and policy
                self.policy_eval_improve(prev_s, prev_a, prev_r, current_state, action, done)

            iter_episode += 1

            # print('end of loop: ',current_state)
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time  # Calculate total running time
        print(f"Total running time: {total_time:.2f} seconds")
            # raise NotImplementedError

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))
        # raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        batch_size = min(self.sample_batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        return batch
        # raise NotImplementedError

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        
        if is_done:
            target = r
        else:
            max_array = self.get_max_state_values()
            target = r + self.discount_factor * max_array[s2]
        td_error = target - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error
        # Update the policy for state s using epsilon-greedy
        policy_indices = self.get_policy_index()    
        best_action = policy_indices[s]
        action_probabilities = np.ones(self.action_space) * (self.epsilon / self.action_space)
        action_probabilities[best_action] += (1.0 - self.epsilon)
        self.policy[s] = action_probabilities
        # raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        start_time = time.time()
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        
        transition_count = 0
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            is_done = False
            while not is_done:
                action_probs = self.policy[current_state]
                action = np.random.choice(self.action_space, p=action_probs)
                next_state, reward, is_done = self.grid_world.step(action)
                # Add transition to replay buffer
                self.add_buffer(current_state, action, reward, next_state, is_done)
                transition_count += 1

                # Every update_frequency steps, sample a batch and update Q-values
                if transition_count % self.update_frequency == 0:
                    batch = self.sample_batch()
                    for s, a, r, s2, done in batch:
                        self.policy_eval_improve(s, a, r, s2, done)
                current_state = next_state
            iter_episode += 1
            # print('iter:',iter_episode)
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time  # Calculate total running time
        print(f"Total running time: {total_time:.2f} seconds")
            