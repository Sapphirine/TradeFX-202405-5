from gym import Env
from gym import spaces
import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
import tempfile
# Suppress all warnings
warnings.filterwarnings("ignore")

class ForexEnvironment(Env):
    # This initializes the ForexEnvironment object
    def __init__(self, initial_value=10000, observations = None,
                 max_steps=100, random_split=True, window_size=5,
                 transaction_fee=0.0001, slippage = 0.0005,
                 order_fraction = 1, reward_function = None,
                 price_column = 'Close', hold=True, diversity_penalty=1):

        self.DEBUG = False

        super(ForexEnvironment, self).__init__()
        self.initial_value = initial_value  # Initial balance
        self.observations = observations  # A dataframe containing price_column
        self.window_size = window_size  # Window size to feed model
        self.balance = initial_value  # Balance = initial_value
        self.shares = 0  # Fractional shares
        self.random_split = random_split # Force the agent to take a random initial market position 
        self.max_steps = max_steps  # Max number of steps to take
        self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1))  # Initialize current step randomly
        self.transaction_fee = transaction_fee  # Fees for each transaction
        self.slippage = slippage  # Maximum slippage coefficient, real coefficient is determined randomly on the interval +/-slippage)
        self.order_fraction = order_fraction  # Ratio of maximum possible order to execute  
        self.price_column = price_column # Name of the column containing price data
        self.rewards = list([i for i in range(window_size)])
        self.action_history = list([0 for i in range(window_size)])
        self.hold = hold # Whether to include hold in the action space
        self.diversity_penalty = diversity_penalty # Reduces reward gaming by penalizing agents who mostly hold their position

        # Check data validity
        if self.observations is None:
            raise ValueError("Observation missing.")
        if len(self.observations) <= 2*self.window_size:
            raise ValueError("Observation data must be longer than double of window size.")


        # Define the action and observation spaces
        num_columns = self.observations.shape[1] - 1  # Exclude the price column
        # 3 actions (buy, sell, hold) or 2 actions (buy, sell)
        if self.hold:
            self.action_space = spaces.Discrete(3) 
        else:
            self.action_space = spaces.Discrete(2) 
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.window_size, num_columns), dtype=np.float32)

        # Reward function setup
        if reward_function is not None: 
            self._get_reward = reward_function 


    # Reset the state of the environment
    def reset(self, start_index = None):
        # Randomly starts index
        if start_index is None:
            self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1))  # Initialize current step randomly
        elif start_index > 1 + 2*self.window_size or start_index < len(self.observations) - self.window_size - self.max_steps:
            raise ValueError('Initial step must be between (1 + 2*window_size, len(observations) - window_size - max_steps).')
        else: self.current_step = start_index

        # Initialize trading position randomly
        if self.random_split:
            # Set vals to zero so that portfolio value is calculated solely on shares value,
            # This allows us to make up the difference in the balance so that portfolio value equals initial balance
            self.balance = 0
            self.shares = 0
            # Split is percentage of assets to be owned as initial shares 
            split = np.random.random()
            current_price = self.observations.iloc[self.current_step][self.price_column]
            self.shares = (self.initial_value*split) / current_price
            self.balance = self.initial_value - self.calculate_portfolio_value(current_price=current_price)
        else:
            self.balance = self.initial_value  # Reset balance
            self.shares = 0  # Reset shares

        return self._get_observation()  # Return observation


    # Execute a trading action in the environment and observe the outcome.
    def step(self, action):

        self.action_history.pop(0)
        self.action_history.append(action)

        current_price = self.observations.iloc[self.current_step][self.price_column]
        total_fees = self.transaction_fee + np.random.uniform(-self.slippage, self.slippage)
        if action == 0:  # Buy
                if self.balance > 0:
                    # Calculate the number of shares we can afford to buy
                    afford_shares = self.order_fraction * (self.balance / (current_price * (1 + total_fees)))
                    # Execute the trade
                    self.shares += afford_shares
                    self.balance -= afford_shares * (current_price * (1 + total_fees))

        elif action == 1:  # Sell
                if self.shares > 0:
                    # Calculate the number of shares we can afford to sell
                    sell_shares = self.order_fraction * self.shares  # Limit the maximum number of shares to sell based on the order fraction
                    self.balance += sell_shares * (current_price * (1 - total_fees))
                    self.shares -= sell_shares

        self.current_step += 1
        done = self.current_step > self.max_steps
        # Return a tuple containing the next observation, reward, done flag, and additional info.
        return self._get_observation(), self._get_reward(action), done, {}


    # Get the current market data from the environment.
    def _get_observation(self):
        start_index = self.current_step - self.window_size
        end_index = self.current_step
        window_data = self.observations.iloc[start_index-self.window_size:end_index].drop(self.price_column, axis = 1)  # Exclude the price column
    
        # Min-Max scale the windowed data
        scaled_df = pd.DataFrame({col: (window_data[col] - window_data[col].min()) / (window_data[col].max() - window_data[col].min() + 1e-8) for col in window_data.columns})
        return scaled_df.values[self.window_size:]


    # Calculate the reward received after executing an action.
    def _get_reward(self, action):
        time_shift = 1
        prev_price = self.observations.iloc[self.current_step - 2 + time_shift][self.price_column]
        current_price = self.observations.iloc[self.current_step - 1 + time_shift][self.price_column]

        prev_portfolio_value = self.calculate_portfolio_value(current_price=prev_price)
        current_portfolio_value = self.calculate_portfolio_value(current_price=current_price)

        # Calculate immediate profit/loss from the current action
        if action == 0:  # Buy
            immediate_profit = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        elif action == 1:  # Sell
            immediate_profit = (prev_portfolio_value - current_portfolio_value) / current_portfolio_value
        else:  # Hold
            immediate_profit = 0

        # Calculate diversity penalty
        diversity_penalty = 0.0
        window_size = self.window_size  # Define the window size

        # Track the frequency or distribution of actions over the sliding window
        action_history = self.action_history[-window_size:]
        action_counts = np.bincount(action_history)
        action_probabilities = action_counts / window_size

        valid_probabilities = action_probabilities[action_probabilities != 0]
        entropy = -np.sum(valid_probabilities * np.log(valid_probabilities)) if len(valid_probabilities) > 0 else 0

        # Apply penalty based on entropy (penalize low entropy)
        diversity_penalty = self.diversity_penalty * (1 - entropy)  # Adjust penalty value as needed

        reward = immediate_profit - diversity_penalty

        return reward



    def calculate_portfolio_value(self, current_price):
        shares_value = current_price*self.shares
        trading_fee = shares_value * self.transaction_fee
        return self.balance + shares_value - trading_fee


    # Evaluate the trading strategy over a specified frame length.
    def evaluate(self, frame_length, start_index = None, model=None, deterministic=False, marker_size=20, init_balance=10000, init_shares=0, verbose = 1, figsize=(14,6)):
        # frame_length is the length of the evaluation frame.
        # start_index is the tarting index for evaluation.

        # Backup current state to restore after evaluation
        prev_value = self.balance 
        prev_shares = self.shares
        prev_step = self.current_step 

        # Reset the environment for evaluation
        self.reset()

        #  # Select a random subset of the data for evaluation if no start_index is specified
        if start_index is None:
            self.current_step = np.random.randint(1 + 2*self.window_size, len(self.observations) - self.max_steps - 2*(self.window_size + 1) - frame_length)  # Initialize current step randomly
        elif start_index < 1 + 2*self.window_size or start_index > len(self.observations) - self.max_steps - 2*(self.window_size + 1) - frame_length:
            raise ValueError('Initial step should be on the interval (1 + 2*window_size, len(observations) - max_steps - 2*window_size - frame_length. \n)')
        else: self.current_step = start_index
        eval_start_index = self.current_step + self.window_size
        eval_end_index = eval_start_index + frame_length
        eval_data = self.observations.iloc[eval_start_index:eval_end_index].reset_index(drop=True)

        # Randomly assign initial position split
        t0balance = init_balance
        t0shares = init_shares

        # Randomly assign initial position split if random_split is enabled
        if self.random_split:
            # Initialize vals to zero so that portfolio value is calculated solely on shares value,
            # this allows us to make up the different in the balance so that portfolio value equals initial balance
            self.balance = 0
            self.shares = 0
            # Split is percentage of assets to be owned as initial shares 
            split = np.random.random()
            current_price = self.observations.iloc[self.current_step][self.price_column]
            self.shares = (init_balance*split) / current_price
            self.balance = init_balance - self.calculate_portfolio_value(current_price=current_price)
        else:
            self.balance = init_balance  # Reset balance
            self.shares = init_shares  # Reset shares

        t0balance = self.balance
        t0shares = self.shares
        

        fig = None

        # Price subplot
        fig, axs = plt.subplots(2, figsize=figsize, sharex=True, height_ratios=[3,1])
        axs[0].plot(range(0,len(eval_data[self.price_column])), eval_data[self.price_column], label=('Close Price'))
        plt.suptitle('Trading Evaluation')
        plt.xlabel('Time Step')
        axs[0].set_ylabel('Price')
        axs[0].grid(visible=True, alpha = 0.5)


        profit = 0.0  # Accumulated profit
        observation = self._get_observation()  # Get the initial observation
        current_price = None
        rewards = []
        portfolio_vals = []

        for index, row in eval_data.iterrows():
            # Perform action using the trained model
            action, _ = model.predict(observation, deterministic=deterministic)
            action = int(action)

            # Step through the environment with the selected action
            next_observation, reward, done, _ = self.step(action)

            # Accumulate the rewards
            current_price = row[self.price_column]
            rewards.append(self._get_reward(action))
            portfolio_vals.append(self.calculate_portfolio_value(current_price=current_price))

            # Price subplot
            # Plot marker based on action
            if action == 0:  # Buy
                axs[0].scatter(index, current_price, color='green', marker='o', s=marker_size)
            elif action == 1:  # Sell
                axs[0].scatter(index, current_price, color='red', marker='o', s=marker_size)
            elif action==2: axs[0].scatter(index, current_price, color='blue', marker='o', s=marker_size)

            # Update the observation for the next step
            observation = next_observation

        # Portfolio value subplot
        axs[1].set_ylabel('Portfolio Value')
        axs[1].plot(portfolio_vals, color='#15ab5b', lw=2)
        axs[1].grid(visible=True, alpha = 0.5)

        temp_filename = next(tempfile._get_candidate_names())

        temp_filepath = os.path.join("./static/", temp_filename + ".png")

        temp_filename = temp_filename + ".png"

        print('Saving plot to: ' + temp_filepath)
        plt.savefig(temp_filepath)

        # Print evaluation metrics
        if verbose>0:
            print("Evaluation Metrics:  ")
            print('Initial value:       ' + format(portfolio_vals[0],'.2f'))
            print('Initial balance:     ' + format(t0balance, '.2f'))
            print('Initial shares:      ' + format(t0shares, '.2f'))
            if self.random_split:
                print('Initial split:       ' + format(1-split,'.2f'))
            print("Final value:         " + format(portfolio_vals[-1], '.2f'))
            print("Profit:              " + format(portfolio_vals[-1]-portfolio_vals[0],'.2f'))
            print("Return Rate:         " + format((portfolio_vals[-1]/portfolio_vals[0] - 1 ),'.2%'))
            print("Avg reward:          " + format(sum(rewards)/len(rewards),'.2f'))
            print("Max reward:          " + format(max(rewards),'.2f'))
            print("Min reward:          " + format(min(rewards),'.2f'))
            print("Cumulative reward:   " + format(sum(rewards),'.2f'))

        # Restore the environment state to its original values
        self.balance = prev_value
        self.shares = prev_shares
        self.current_step = prev_step
        if self.random_split:
            return temp_filename, {"initial_value":portfolio_vals[0], "initial_balance":t0balance, "initial_shares":t0shares, "initial_split":1-split, "final_value":portfolio_vals[-1], "profit":portfolio_vals[-1]-portfolio_vals[0], "return_rate":portfolio_vals[-1]/portfolio_vals[0] - 1, "avg_reward":sum(rewards)/len(rewards), "max_reward":max(rewards), "min_reward":min(rewards), "cumulative_reward":sum(rewards)}
        else:
            return temp_filename, {"initial_value":portfolio_vals[0], "initial_balance":t0balance, "initial_shares":t0shares, "final_value":portfolio_vals[-1], "profit":portfolio_vals[-1]-portfolio_vals[0], "return_rate":portfolio_vals[-1]/portfolio_vals[0] - 1, "avg_reward":sum(rewards)/len(rewards), "max_reward":max(rewards), "min_reward":min(rewards), "cumulative_reward":sum(rewards)}