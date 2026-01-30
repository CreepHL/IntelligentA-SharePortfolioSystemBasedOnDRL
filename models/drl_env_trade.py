import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt

# 100 shares per trade
HMAX_NORMALIZE = 100
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
# avoid gradient explosion
REWARD_SCALING = 1e-4
INITIAL_ACCOUNT_BALANCE = 1000000

feature_columns = [
    'Open', 'Close', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'MA20',
    'RSI', 'MACD', 'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band',
    'Relative_Performance', 'ATR', 'predict_percentages'
]


class StockEnvTrade(gym.Env):
    def __init__(self, df, hold, day=0, initial=True, previous_state=[], model_name='', iteration=''):
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        self.stock_dim = len([col for col in df.columns if 'Tic' in col])
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
        # 动态股票相关数据
        # 状态空间维度: balance(1) + prices(stock_dim) + holdings(stock_dim) + features(stock_dim * num_features)
        state_dim = 1 + self.stock_dim * 2 + self.stock_dim * len(feature_columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        # 获取当前日期的数据
        self.data = self.df.iloc[[self.day]]
        self.terminal = False
        self.state = self.get_state(hold)
        self.reward = 0
        self.cost = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self.np_random, seed = seeding.np_random(seed=None)
        self.model_name = model_name
        self.iteration = iteration

    def get_state(self, hold):
        """构建状态向量"""
        # 余额
        # 计算初始余额：capital - sum(每只股票价格 × 持仓数量)
        balance = [hold['capital'] - sum(price_quantity[0] * price_quantity[1]
                                         for price_quantity in hold['stocks'].values())] \
            if hasattr(hold, 'capital') else [INITIAL_ACCOUNT_BALANCE]
        # 预测价格
        prediction_cols = [col for col in self.data.columns if 'prediction' in col.lower()]
        prices = []
        for col in prediction_cols:
            prices.extend(self.data[col].values.tolist())
        # 获取持仓信息：对于df['tic']中每个股票代码，从hold中获取对应持仓量，不存在则为0
        holdings = []
        for tic in [col for col in self.data.columns if 'Tic' in col]:  # 遍历当前数据中的所有股票代码
            if self.data[tic][0] in hold['stocks']:
                holding_quantity = hold['stocks'][tic][1]  # 获取持仓数量
            else:
                holding_quantity = 0  # 没有持仓则为0
            holdings.append(holding_quantity)
        # 技术指标
        features = []
        for feature in feature_columns:
            for col in [col for col in self.data.columns if feature in col]:
                if col in self.data.columns:
                    features.extend(self.data[col].values.tolist())
                else:
                    features.extend([0] * self.stock_dim)
        return balance + prices + holdings + features

    def sell(self, index, action):
        if self.state[index + self.stock_dim + 1] > 0:
            # update balance
            self.state[0] += \
                self.state[index + 1] * min(abs(action), self.state[index + self.stock_dim + 1]) * \
                (1 - TRANSACTION_FEE_PERCENT)
            self.state[index + self.stock_dim + 1] -= min(abs(action), self.state[index + self.stock_dim + 1])
            self.cost += self.state[index + 1] * min(abs(action),
                                                     self.state[index + self.stock_dim + 1]) * TRANSACTION_FEE_PERCENT
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

        self.state[index + self.stock_dim + 1] += min(available_amount, action)

        self.cost += self.state[index + 1] * min(available_amount, action) * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                    self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])) -
                                           self.asset_memory[0]))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            print("Sharpe: ", sharpe)
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            # df_rewards.to_csv('results/account_rewards_train.csv')
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            # actions = (actions.astype(int))

            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                    self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
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
            self.data = self.df.iloc[[self.day]]
            # print("stock_shares:{}".format(self.state[29:]))
            features = []
            for feature in feature_columns:
                for col in [col for col in self.data.columns if feature in col]:
                    if col in self.data.columns:
                        features.extend(self.data[col].values.tolist())
                    else:
                        features.extend([0] * self.stock_dim)

            prediction_cols = [col for col in self.data.columns if 'prediction' in col.lower()]
            prices = []
            for col in prediction_cols:
                prices.extend(self.data[col].values.tolist())
            self.state = [self.state[0]] + prices + \
                         list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + \
                         features

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.stock_dim + 1)]) * np.array(
                                  self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))
            self.reward = end_total_asset - begin_total_asset
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.iloc[[self.day]]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            # initiate state
            self.state = self.get_state()
            # iteration += 1

        else:
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(self.stock_dim + 1)]) * np.array(
                                       self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]))
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.df.iloc[[self.day]]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            features = []
            for feature in feature_columns:
                for col in [col for col in self.data.columns if feature in col]:
                    if col in self.data.columns:
                        features.extend(self.data[col].values.tolist())
                    else:
                        features.extend([0] * self.stock_dim)

            prediction_cols = [col for col in self.data.columns if 'prediction' in col.lower()]
            prices = []
            for col in prediction_cols:
                prices.extend(self.data[col].values.tolist())
            self.state = [self.previous_state[0]] + prices + \
                         list(self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + \
                         features
        return self.state

    def render(self, mode='human'):
        return self.state
