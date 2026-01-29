import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt

# 100 shares per trade
HMAX_NORMALIZE = 100
STOCK_DIM = 1
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
# avoid gradient explosion
REWARD_SCALING = 1e-4

class StockEnvTrain(gym.Env):
    def __init__(self, df, day=0):
        self.day = day
        self.df = df
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # 10支股票相关数据
        self.observation_space = spaces.Box(high=np.inf, shape=())
        self.data = self.df.loc[]
        self.terminal = False

        self.state = [INITIAL_ACCOUNT_BALANCE] + self.data.... + [初始持仓]
        self.reward = 0
        self.cost = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self.np_random, seed = seeding.np_random(seed=None)

    def sell(self, index, action):
        if self.state[index+1] > 0:
            # update balance
            self.state[0] += \
                self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * \
                (1 - TRANSACTION_FEE_PERCENT)

            self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
            self.cost += self.state[index + 1] * min(abs(action), self.state[index + STOCK_DIM + 1]) * \
                         TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass


    def buy(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))

        # update balance
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                         (1 + TRANSACTION_FEE_PERCENT)

        self.state[index + STOCK_DIM + 1] += min(available_amount, action)

        self.cost += self.state[index + 1] * min(available_amount, action) * \
                     TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            # print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            # print("total_cost: ", self.cost)
            # print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.to_csv('results/account_rewards_train.csv')
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            # actions = (actions.astype(int))

            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self.sell(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self.buy(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            # print("stock_shares:{}".format(self.state[29:]))
            self.state = [self.state[0]] + \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]) + \
                         self.data.macd.values.tolist() + \
                         self.data.rsi.values.tolist() + \
                         self.data.cci.values.tolist() + \
                         self.data.adx.values.tolist()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))
            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}


    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # iteration += 1
        return self.state


    def render(self, mode='human'):
        return self.state
