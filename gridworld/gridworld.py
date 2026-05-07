import numpy as np

# build gridworld environment
class Gridworld:
    def __init__(self, size_x=4, size_y=4, goal=None, slip_prob=0.0):
        self.size_x = size_x
        self.size_y = size_y
        # States
        self.states = [(x, y) for x in range(size_x) for y in range(size_y)]
        # Initial State
        self.initial_state = (0, 0)
        # Terminal States
        self.terminal_states = [(size_x - 1, size_y - 1)] if goal is None else [goal]
        self.slip_prob = slip_prob

        # Actions
        self.actions = ['up', 'down', 'left', 'right']
        # Transition Model
        self.transition_model = self.stochastic_transitions
        # Reward Function
        self.reward_function = self.simple_reward
        # Discount Factor
        self.discount_factor = 0.9

    def deterministic_transitions(self, state, action):
        # Deterministic transition model
        if state in self.terminal_states:
            return {state: 1.0}  # No transitions from terminal states

        if action == 'up':
            candidate = (state[0] - 1, state[1])
           
        elif action == 'down':
            candidate = (state[0] + 1, state[1])
  
        elif action == 'left':
            candidate = (state[0], state[1] - 1)
      
        elif action == 'right':
            candidate = (state[0], state[1] + 1)
       
        else:
            candidate = state

        x, y = candidate
        if 0 <= x < self.size_x and 0 <= y < self.size_y:
            return {candidate: 1.0}
        else:
            return {state: 1.0}

    def stochastic_transitions(self, state, action):
        # Stochastic transition fuction that returns a distribution over next states
        if state in self.terminal_states:
            return {state: 1.0}  # No transitions from terminal states

        next_states = {}
        for a in self.actions:
            if a == action:
                next_state = self._deterministic_next(state, a)
                next_states[next_state] = next_states.get(next_state, 0) + (1 - self.slip_prob)
            else:
                next_state = self._deterministic_next(state, a)
                next_states[next_state] = next_states.get(next_state, 0) + self.slip_prob / (len(self.actions) - 1)

        return next_states

    def simple_reward(self, state, action, next_state):
        if next_state in self.terminal_states:
            return 1  # Reward for reaching terminal state
        else:
            return -0.1  # Small negative reward for each step taken

    def value_iteration(self, threshold=0.0001):
        converged = False
        U = {state: 0 for state in self.states}  # Initialize utilities

        while not converged: 
            U_new = {}
            for state in self.states:
                if state in self.terminal_states:
                    U_new[state] = 0
                    continue
                
                # Compute Bellman Update

                U_new[state] = max(
                sum(
                    prob * (self.reward_function(state, action, next_state) + self.discount_factor * U[next_state])
                    for next_state, prob in self.transition_model(state, action).items()
                )
                for action in self.actions
            )
                                   
            # Check for convergence
            if max(abs(U_new[state] - U[state]) for state in self.states) < threshold:
                converged = True
            U = U_new
        return U

    def policy_extraction(self, U):

        policy = {}
        for state in self.states:
            if state in self.terminal_states:
                policy[state] = None
                continue

            best_action = None
            best_value =  float('-inf')

            for action in self.actions:
                next_states = self.transition_model(state, action)
                value = sum(
                    prob * (self.reward_function(state, action, next_state) + self.discount_factor * U[next_state])
                    for next_state, prob in next_states.items()
                )
                if value > best_value:
                    best_value = value
                    best_action = action

            policy[state] = best_action
        return policy

    def policy_iteration(self, threshold=0.0001):
        # Initialize a random policy
        policy = {state: np.random.choice(self.actions) for state in self.states if state not in self.terminal_states}
        policy.update({state: None for state in self.terminal_states})  # No actions for terminal states

        while True:
            # Policy Evaluation
            U = {state: 0 for state in self.states}  # Initialize utilities
            while True:
                U_new = {}
                for state in self.states:
                    if state in self.terminal_states:
                        U_new[state] = 0
                        continue
                    
                    action = policy[state]
                    U_new[state] = sum(
                        prob * (self.reward_function(state, action, next_state) + self.discount_factor * U[next_state])
                        for next_state, prob in self.transition_model(state, action).items()
                    )
                if max(abs(U_new[state] - U[state]) for state in self.states) < threshold:
                    break
                U = U_new

            new_policy = self.policy_extraction(U)
            policy_stable = all(
                policy[s] == new_policy[s] 
                for s in self.states 
                if s not in self.terminal_states
            )
            policy = new_policy
            if policy_stable:
                break

        return policy
        


    def trace_generation(self, policy, start_state=None, max_steps=100):
        if start_state is None:
            start_state = self.initial_state
        trace = []
        current_state = start_state 
        for _ in range(max_steps):
            action = policy[current_state]
            if action is None:  # Reached terminal state
                break
            next_state = self.sample_transition(current_state, action)
            reward = self.reward_function(current_state, action, next_state)
            trace.append((current_state, action, reward, next_state))
            current_state = next_state
        return trace

    def sample_transition(self, state, action):
        distribution = self.transition_model(state, action)
        next_states = list(distribution.keys())
        probabilities = list(distribution.values())
        idx = np.random.choice(len(next_states), p=probabilities)
        return next_states[idx]
   

    def _deterministic_next(self, state, action):
        return next(iter(self.deterministic_transitions(state, action).keys()))


    def trace_generation_random(self, start_state=None, max_steps=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if start_state is None:
            start_state = self.initial_state
        trace = []
        current_state = start_state
        for _ in range(max_steps):
            if current_state in self.terminal_states:
                break
            action = rng.choice(self.actions)  # fresh random action per step
            next_state = self.sample_transition(current_state, action)
            reward = self.reward_function(current_state, action, next_state)
            trace.append((current_state, action, reward, next_state))
            current_state = next_state
        return trace

    def trace_generation_eps_greedy(self, optimal_policy, epsilon, start_state=None, max_steps=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if start_state is None:
            start_state = self.initial_state
        trace = []
        current_state = start_state
        for _ in range(max_steps):
            if current_state in self.terminal_states:
                break
            if rng.random() < epsilon:
                action = rng.choice(self.actions)
            else:
                action = optimal_policy[current_state]
            next_state = self.sample_transition(current_state, action)
            reward = self.reward_function(current_state, action, next_state)
            trace.append((current_state, action, reward, next_state))
            current_state = next_state
        return trace
    

def generate_dataset(num_mdps=20, traces_per_policy=50, max_steps=100, seed=None):
    rng = np.random.default_rng(seed)
    mdps_metadata = []
    traces = []
    
    for mdp_idx in range(num_mdps):
        # generate one MDP from the family
        mdp_seed = int(rng.integers(0, 2**31))
        mdp = random_mdp(seed=mdp_seed)
        
        mdps_metadata.append({
            'mdp_idx': mdp_idx,
            'mdp_seed': mdp_seed,
            'size_x': mdp.size_x,
            'size_y': mdp.size_y,
            'goal': mdp.terminal_states[0],
            'slip_prob': mdp.slip_prob,
           
        })
        
        # compute optimal policy once
        U = mdp.value_iteration()
        optimal_policy = mdp.policy_extraction(U)
        
        # generate traces for each policy type
        for _ in range(traces_per_policy):
            trace_seed = int(rng.integers(0, 2**31))
            trace_rng = np.random.default_rng(trace_seed)
            
            opt_trace = mdp.trace_generation(optimal_policy, max_steps=max_steps)
            eps_trace = mdp.trace_generation_eps_greedy(optimal_policy, epsilon=0.1, max_steps=max_steps, rng=trace_rng)
            rand_trace = mdp.trace_generation_random(max_steps=max_steps, rng=trace_rng)
            
            traces.append({'mdp_idx': mdp_idx, 'policy_type': 'optimal', 'trace': opt_trace})
            traces.append({'mdp_idx': mdp_idx, 'policy_type': 'eps_greedy_01', 'trace': eps_trace})
            traces.append({'mdp_idx': mdp_idx, 'policy_type': 'random', 'trace': rand_trace})
    
    return {'mdps': mdps_metadata, 'traces': traces}

def build_vocab(max_grid_size, actions):
    vocab = {}
    # state coordinates
    for i in range(max_grid_size):
        vocab[f'coord_{i}'] = len(vocab)
    # actions
    for a in actions:
        vocab[f'action_{a}'] = len(vocab)
    # rewards (you have exactly 2 distinct values in your dataset)
    vocab['reward_step'] = len(vocab)  # for -0.1
    vocab['reward_goal'] = len(vocab)  # for 1
    # special
    vocab['<eos>'] = len(vocab)
    return vocab

def tokenize_trace(trace, vocab):
    tokens = []
    for step in trace:
        state, action, reward, next_state = step
        tokens.append(vocab[f'coord_{state[0]}'])
        tokens.append(vocab[f'coord_{state[1]}'])
        tokens.append(vocab[f'action_{action}'])
        tokens.append(vocab['reward_step'] if reward < 0 else vocab['reward_goal'])
        tokens.append(vocab[f'coord_{next_state[0]}'])
        tokens.append(vocab[f'coord_{next_state[1]}'])
    tokens.append(vocab['<eos>'])
    return tokens


def detokenize(tokens, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    trace = []
    i = 0
    while i < len(tokens):
        if tokens[i] == vocab['<eos>']:
            break
        state_x = inv_vocab[tokens[i]].split('_')[1]
        state_y = inv_vocab[tokens[i+1]].split('_')[1]
        action = inv_vocab[tokens[i+2]].split('_')[1]
        reward_token = inv_vocab[tokens[i+3]]
        reward = -0.1 if reward_token == 'reward_step' else 1
        next_x = inv_vocab[tokens[i+4]].split('_')[1]
        next_y = inv_vocab[tokens[i+5]].split('_')[1]
        trace.append(((int(state_x), int(state_y)), action, reward, (int(next_x), int(next_y))))
        i += 6
    return trace

def random_mdp(seed=None):
    rng = np.random.default_rng(seed)
    size_x = int(rng.integers(3, 6))
    size_y = int(rng.integers(3, 6))
    candidates = [(x, y) for x in range(size_x) for y in range(size_y) if (x, y) != (0, 0)]
    goal_state = candidates[rng.integers(len(candidates))]
    slip_prob = float(rng.uniform(0, 0.3))
    return Gridworld(size_x=size_x, size_y=size_y, goal=goal_state, slip_prob=slip_prob)