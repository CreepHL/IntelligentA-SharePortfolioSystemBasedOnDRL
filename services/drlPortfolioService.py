import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from models.drl_env_trade import StockEnvTrade
from models.drl_env_train import StockEnvTrain


def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df['Date'] > start) & (df['Date'] <= end)]
    #data  = data[final_columns]
    data.index = data.Date.factorize()[0]
    return data


def train_ppo(env_train, model_name, timesteps=50000):
    """PPO model"""

    start = time.time()
    model = PPO('MlpPolicy', env_train, ent_coef = 0.005, batch_size=256, n_steps=2048, n_epochs=10)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"./models/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   initial):
    ### make a prediction based on trained model###

    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('./output/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def run_drl_portfolio(stock_df, unique_trade_date, rebalance, validation, hold):
    last_state_ensemble = []
    ppo_sharpe_list = []
    start = time.time()
    for i in range(rebalance + validation, len(unique_trade_date), rebalance):
        if i - rebalance - validation == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False
        train = data_split(stock_df, start=pd.to_datetime('2009-01-01'), end=unique_trade_date[i - rebalance - validation])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train, hold=hold)])
        initial = True
        # validation = data_split(stock_df, start=unique_trade_date[i - rebalance - validation], end=unique_trade_date[i - rebalance])
        # env_val = DummyVecEnv([lambda: StockEnvValidation(validation, i - rebalance)])
        # obs_val = env_val.reset()
        model_ppo = train_ppo(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
        last_state_ensemble = DRL_prediction(df=stock_df, model=model_ppo, name="ppo",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance,
                                             initial=initial)
        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
