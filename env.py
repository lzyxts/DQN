
import numpy as np
import itertools

class MarketEnv:
    def __init__(self, data, nstock, initial_investment=50000, tran_cost=0.0):
        self.data = data  # Trading and Econ data
        self.n_step, self.n_stock = self.data.shape[0], nstock  # Declare variables with input data structure
        self.stock_prices = self.data[:, :self.n_stock]  # Extract stock prices from the dataset
        self.daily_rets = self.data[:, self.n_stock:self.n_stock * 2]  # Extract daily returns from the dataset
        self.daily_rets = np.hstack((self.daily_rets, np.zeros((self.n_step, 1))))  # Adding zero return for cash
        self.tran_cost = tran_cost  # Transaction cost
        self.initial_investment = initial_investment  # Initial investment

        # Calculate the size of state dimension, position holdings + market data input
        self.state_dim = self.n_stock + self.data.shape[1] + 1

        # Initiate other attributes
        self.cur_step = None
        self.cur_holdings = None
        self.stock_price = None
        self.daily_ret = None
        self.cur_action_idx = None

        # Generate a list of possible combinations, exclude [0, 0, 0] here
        self.selections = list(map(list, itertools.product([0, 1], repeat=self.n_stock + 1)))[1:]
        # Convert to % of the target portfolio allocation
        self.action_space = np.array([[val / sum(combo) for val in combo] for combo in self.selections])
        # Should equal to 2^(N+1)-1
        self.action_space_dim = len(self.action_space)

        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        self.cur_step = 0
        self.cur_holdings = np.array([0.0] * self.n_stock + [self.initial_investment])
        self.stock_price = self.stock_prices[self.cur_step]
        self.daily_ret = self.daily_rets[self.cur_step]
        self.portfolio_rets = np.zeros(self.n_step)
        self.portfolio_rets[0] = 0.0
        self.cur_action_idx = 0
        return self.get_state()

    # Transaction cost measured the actual dollar amount, 2 basis point would be 0.0002
    def step(self, action, verbose=False):
        # Get current value before performing the action
        prev_eod_val = self.get_val()

        if verbose:
            print(f'cur_step: {self.cur_step}, position before trade: {self.cur_holdings}, '
                  f'port_val before trade: {prev_eod_val}, taking action: {self.action_to_str(action)}')

        # Perform the trade and update holdings
        self.cur_holdings = self._trade(action, self.cur_holdings, self.tran_cost)

        if verbose:
            print(f'after trades and relevant transaction cost: {self.cur_holdings} to end the day')

        # Update price, i.e. go to the next day
        self.cur_step += 1
        self.stock_price = self.stock_prices[self.cur_step]
        self.daily_ret = self.daily_rets[self.cur_step]
        self.cur_holdings *= 1 + self.daily_ret
        self.cur_action_idx = action
        # Get the new portfolio value after taking the action
        cur_val = self.get_val()

        # Calculate the portfolio value change as the reward
        reward = self.get_reward(cur_val, prev_eod_val)

        if verbose:
            print(f'EOD pos: {self.cur_holdings}, EOD port val: {cur_val}, reward: {reward}')

        # done if we reach the end of the data
        done = self.cur_step == (self.n_step - 1)

        # Store information about the current situation as needed
        info = {'cur_step': self.cur_step, 'cur_val': cur_val}

        return self.get_state(), reward, done, info

    def get_reward(self, cur_val, prev_eod_val):
        # is the change in port val
        # Other options also exist, e.g. daily % returns, Sharpe ratio, Sortino ratio, etc.
        return (cur_val - prev_eod_val)

    def get_state(self):
        # Get the current state of the environment
        return np.concatenate([self.cur_holdings, self.data[self.cur_step]])

    def get_val(self):
        # The valuation is simply the sum of all holdings in each asset
        return sum(self.cur_holdings)

    def action_to_str(self, action):
        # Convert action index to action vector (portfolio allocation)
        return self.action_space[action]

    def _trade(self, action, cur_pos, trans_cost):
        # Get the target allocation
        action_vec = self.action_space[action]

        # assuming we sell everything at the closing price, then buy the target allocation at closing price
        # assuming we can purchase fractional shares

        tot_val = sum(cur_pos)  # Total portfolio value available to re-allocate
        target_allocation = tot_val * action_vec  # Total target value in each asset after rebalance

        delta = (target_allocation - cur_pos)[:self.n_stock]  # Determine the assets to sell
        tot_trans_cost = sum(delta[delta < 0]) * trans_cost  # Compute the total transaction cost
        tot_val += tot_trans_cost  # Compute the total transaction cost

        return tot_val * action_vec  # Allocate the portfolio value after taking into account the transaction cost
