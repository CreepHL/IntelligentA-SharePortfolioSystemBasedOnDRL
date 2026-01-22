import asyncio
import os
import pickle
from datetime import datetime, timedelta
from typing import List

import numpy
import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from models.LSTM_model import LSTMMain
from services.technicalService import get_stock_data
from tools.visualization import plot_cumulative_earnings, plot_accuracy_comparison, plot_stock_prediction

# # 必要参数定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备,如果NVIDIA GPU已配置，会自动使用GPU训练
# train_ratio = 0.8  # 训练集比例
# val_ratio = 0.1  # 验证集比例
# test_ratio = 0.1  # 测试集比例
# batch_size = 32  # 批大小，若用CPU，建议为1
# input_length = 20  # 每个batch的输入数据长度，多步预测建议长，单步预测建议短
# output_length = 1  # 每个batch的输出数据长度，1为单步预测，1以上为多步预测
# loss_function = 'MSE'  # 损失函数定义
# learning_rate = 0.001  # 基础学习率
# weight_decay = 0.0001  # 权重衰减系数
# num_blocks = 2  # lstm堆叠次数
# dim = 64  # 隐层维度
# interval_length = 0  # 预测数据长度，最长不可以超过总数据条数
# scalar = True  # 是否使用归一化
# scalar_contain_labels = True  # 归一化过程是否包含目标值的历史数据
# target_value = 'CEP'  # 需要预测的列名，可以在excel中查看
# out_date = {}
# test_date = []
# criterion = nn.MSELoss()
# num_epochs = 10

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


def plot_training_loss(ticker, train_losses, val_losses, save_dir):
    """
    绘制训练和验证损失曲线

    参数:
        ticker: 股票代码
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {ticker}')
    plt.legend()
    plt.grid(True)

    loss_dir = os.path.join(save_dir, 'pic/loss')
    os.makedirs(loss_dir, exist_ok=True)
    save_path = os.path.join(loss_dir, f'{ticker}_loss.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


def get_rolling_window_multistep(forecasting_length, interval_length, window_length, features, labels):
    total_timesteps = features.shape[1]
    features = np.asarray(features)
    max_start = total_timesteps - window_length - interval_length - forecasting_length
    if max_start < 0:
        raise ValueError("Not enough data for given window/forecast lengths!")

    output_features, output_labels = [], []

    for start in tqdm.tqdm(range(max_start + 1), desc='Preparing data'):
        # 输入窗口: [start, start + window_length)
        x = features[:, start:start + window_length]  # (num_features, window_length)
        output_features.append(x)

        # 预测窗口: 从 (start + window_length + interval_length) 开始，取 forecasting_length 步
        label_start = start + window_length + interval_length
        y = labels[:, label_start:label_start + forecasting_length]  # (num_targets, forecasting_length)
        output_labels.append(y)

    # 合并为数组
    X = np.stack(output_features, axis=0).astype(np.float32)    # (num_samples, num_features, window_length)
    y = np.stack(output_labels, axis=0).astype(np.float32) # (num_samples, num_targets, forecasting_length)

    return torch.from_numpy(X), torch.from_numpy(y)


def visualize_predictions(ticker, data, predict_result, test_indices, predictions, actual_percentages, save_dir):
    actual_prices = data['Close'].loc[test_indices].values
    predicted_prices = np.array(predictions)

    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    accuracy = 1 - np.mean(np.abs(predicted_prices - actual_prices) / actual_prices)

    metrics = {'rmse': rmse, 'mae': mae, 'accuracy': accuracy}
    plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir)

    return metrics


def train_and_predict_lstm(ticker, data, X, y, save_dir, n_steps=60, num_epochs=100, batch_size=32, learning_rate=0.001):
    # 数据归一化和准备部分
    scaler_y = MinMaxScaler()
    scaler_X = MinMaxScaler()
    scaler_y.fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    X_train, y_train = prepare_data(X_scaled, n_steps)
    y_train = y_scaled[n_steps-1:-1]

    train_per = 0.8
    split_index = int(train_per * len(X_train))
    X_val = X_train[split_index-n_steps+1:]
    y_val = y_train[split_index-n_steps+1:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]

    # PyTorch数据准备
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMMain(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses = []
    val_losses = []

    with tqdm.tqdm(total=num_epochs, desc=f"Training {ticker}", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # 训练和验证循环
            model.train()
            epoch_train_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, targets)
                    epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
            pbar.update(1)
            scheduler.step()

    # 使用可视化工具绘制损失曲线
    plot_training_loss(ticker, train_losses, val_losses, save_dir)

    # 预测
    model.eval()
    predictions = []
    test_indices = []
    predict_percentages = []
    actual_percentages = []

    with torch.no_grad():
        for i in range(1 + split_index, len(X_scaled) + 1):
            x_input = torch.tensor(X_scaled[i - n_steps:i].reshape(1, n_steps, X_train.shape[2]),
                                 dtype=torch.float32).to(device)
            y_pred = model(x_input)
            y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
            predictions.append((1 + y_pred[0][0]) * data['Close'].iloc[i - 2])
            test_indices.append(data.index[i - 1])
            predict_percentages.append(y_pred[0][0] * 100)
            actual_percentages.append(y[i - 1] * 100)

    # 使用可视化工具绘制累积收益率曲线
    plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir)

    predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
    return predict_result, test_indices, predictions, actual_percentages


def predict(ticker_name, stock_data, stock_features, save_dir, epochs=100, batch_size=32, learning_rate=0.001):
    all_predictions_lstm = {}
    prediction_metrics = {}

    print(f"\nProcessing {ticker_name}")
    data = stock_data
    X, y = stock_features

    predict_result, test_indices, predictions, actual_percentages = train_and_predict_lstm(
        ticker_name, data, X, y, save_dir, num_epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    all_predictions_lstm[ticker_name] = predict_result

    metrics = visualize_predictions(ticker_name, data, predict_result, test_indices, predictions, actual_percentages,
                                    save_dir)
    prediction_metrics[ticker_name] = metrics

    save_predictions_with_indices(ticker_name, test_indices, predictions, save_dir)

    # 保存预测指标
    os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)
    metrics_df = pd.DataFrame(prediction_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_metrics.csv'))
    print("\nPrediction metrics summary:")
    print(metrics_df.describe())

    # 使用可视化工具绘制准确度对比图
    plot_accuracy_comparison(prediction_metrics, save_dir)

    # 生成汇总报告
    summary = {
        'Average Accuracy': np.mean([m['accuracy'] * 100 for m in prediction_metrics.values()]),
        'Best Stock': max(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Worst Stock': min(prediction_metrics.items(), key=lambda x: x[1]['accuracy'])[0],
        'Average RMSE': metrics_df['rmse'].mean(),
        'Average MAE': metrics_df['mae'].mean()
    }

    # 保存汇总报告
    with open(os.path.join(save_dir, 'output', f'{ticker_name}_prediction_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f'{key}: {value}\n')

    print("\nPrediction Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return metrics


def save_predictions_with_indices(ticker, test_indices, predictions, save_dir):
    df = pd.DataFrame({
        'Date': test_indices,
        'Prediction': predictions
    })

    file_path = os.path.join(save_dir, 'predictions', f'{ticker}_predictions.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(df, file)

    print(f'Saved predictions for {ticker} to {file_path}')


async def lstm_stock_predict(stock_code: str):
    current_date = datetime.now()
    yesterday = current_date - timedelta(days=1)
    end_date = yesterday.strftime('%Y-%m-%d')
    stock_data = get_stock_data(stock_code, '2016-01-16', end_date)
    features = [
        'Volume', 'Year', 'Month', 'Day', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'Relative_Performance', 'ATR',
        'Close_yes', 'Open_yes', 'High_yes', 'Low_yes'
    ]
    X = stock_data[features].iloc[1:]
    y = stock_data['Close'].pct_change().iloc[1:]
    stock_features = X, y
    predict(stock_code, stock_data, stock_features, 'output')

    # df = history_price(stock_code, '2016-01-16', '2026-01-16')
    # # 收益率作为labels
    # date = pd.to_datetime(df['date'].iloc[1:])
    # labels_ = df['pct_change'].iloc[1:]
    # features = df[['MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'VWAP', 'SMA', 'Std_dev', 'Upper_band', 'Lower_band', 'atr',
    #     'close_yes', 'open_yes', 'high_yes', 'low_yes']].iloc[1:]
    # features, labels = get_rolling_window_multistep(output_length, 0, input_length, features.T, np.expand_dims(labels_, 0))
    # #  构建数据集
    # labels = torch.squeeze(labels, dim=1)
    # split_train_val, split_val_test = int(len(features) * train_ratio), int(len(features) * train_ratio) + int(
    #     len(features) * val_ratio)
    # train_features, train_labels = features[:split_train_val], labels[:split_train_val]
    # val_features, val_labels = features[split_train_val:split_val_test], labels[split_train_val:split_val_test]
    # test_features, test_labels, test_date = features[split_val_test:], labels[split_val_test:], date[split_val_test+input_length:]
    #
    # mean = train_features.mean(dim=(0, 2), keepdim=True)
    # std = train_features.std(dim=(0, 2), keepdim=True) + 1e-6
    # train_features = (train_features - mean) / std
    # val_features = (val_features - mean) / std
    # test_features = (test_features - mean) / std
    # #  数据管道构建，此处采用torch高阶API
    # train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device))
    # train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets, shuffle=True)
    # val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device))
    # val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)
    # test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device))
    # test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)
    # # LSTMMain_model = LSTMMain(input_size=features_num, output_len=output_length,
    # #                           lstm_hidden=dim, lstm_layers=num_blocks, batch_size=batch_size, device=device)
    # LSTMMain_model = LSTMMain(input_size=train_features.shape[1], output_size=output_length, hidden_size=dim,
    #                           num_layers=num_blocks, device=device)
    # epochs = 20
    # # 优化器定义 学习率衰减定义
    # optimizer = torch.optim.Adam(LSTMMain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs // 3, eta_min=0.00001)
    # patience = 3
    # counter = 0
    # val_best = float('inf')
    # for epoch in range(epochs):
    #     LSTMMain_model.train()
    #     train_loss_sum = 0
    #     step = 1
    #     loss_list = []
    #     for step, (feature_, label_) in enumerate(train_Loader):
    #         optimizer.zero_grad()
    #         feature_ = feature_.permute(0, 2, 1)
    #         prediction = LSTMMain_model(feature_)
    #         loss = criterion(prediction, label_)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm(LSTMMain_model.parameters(), 0.15)
    #         optimizer.step()
    #         loss_list.append(loss.item())
    #         train_loss_sum += loss.item()
    #     scheduler.step()
    #     print("epochs = " + str(epoch))
    #     print('train_loss = ' + str(train_loss_sum))
    #     print(loss_list)
    #     numpy.savetxt(f'loss{epoch}.csv', loss_list, delimiter=',')
    #     plot_training_loss(stock_code, loss_list, [], f'./data')
    #     LSTMMain_model.eval()
    #     val_loss_sum = 0
    #     val_step = 1
    #     for val_step, (feature_, label_) in enumerate(val_Loader):
    #         feature_ = feature_.permute(0, 2, 1)
    #         with torch.no_grad():
    #             prediction = LSTMMain_model(feature_)
    #             val_loss = criterion(prediction, label_)
    #         val_loss_sum += val_loss.item()
    #     if val_loss_sum < val_best:
    #         val_best = val_loss_sum
    #         counter = 0
    #         print('val_loss = ' + str(val_loss_sum))
    #     else:
    #         print('val_loss = ' + str(val_loss_sum))
    #         counter += 1
    #         if counter >= patience:
    #             torch.save(LSTMMain_model.state_dict(), f'./models/best_model{stock_code}.pth')
    #             print(f"early stop {epoch + 1}, val_Loss= {val_loss.item(): .4f}")
    #             break
    #
    # print("best val loss = " + str(val_best))
    # print("——————————————————————Training Ends——————————————————————")
    #
    #
    #
    # LSTMMain_model.load_state_dict(torch.load(f'./models/best_model{stock_code}.pth'))  # 调用权重
    # test_loss_sum = 0
    # #  测试集inference
    # print("——————————————————————Testing Starts——————————————————————")
    # for step, (feature_, label_) in enumerate(test_Loader):
    #     feature_ = feature_.permute(0, 2, 1)
    #     with torch.no_grad():
    #         if step == 0:
    #             prediction = LSTMMain_model(feature_)
    #             pre_array = prediction.cpu()
    #             label_array = label_.cpu()
    #             loss = criterion(prediction, label_)
    #             test_loss_sum += loss.item()
    #         else:
    #             prediction = LSTMMain_model(feature_)
    #             pre_array = np.vstack((pre_array, prediction.cpu()))
    #             label_array = np.vstack((label_array, label_.cpu()))
    #             loss = criterion(prediction, label_)
    #             test_loss_sum += loss.item()
    # print("test loss = " + str(test_loss_sum))
    # print("——————————————————————Testing Ends——————————————————————")
    #
    # print("——————————————————————Post-Processing——————————————————————")
    # plt.figure(figsize=(14, 7))
    # plt.plot(test_date, pre_array, 'g')
    # plt.plot(test_date, label_array, "r")
    # plt.legend(["forecast", "actual"], loc='upper right')
    # plt.pause(10)
    # plt.close()
# 数据归一化和准备部分
#     scaler_y = MinMaxScaler()
#     scaler_X = MinMaxScaler()
#     scaler_y.fit(labels_.values.reshape(-1, 1))
#     y_scaled = scaler_y.transform(labels_.values.reshape(-1, 1))
#     X_scaled = scaler_X.fit_transform(features)
#     X_train = prepare_data(features, input_length)
#     y_train = labels_[input_length-1:-1].values.reshape(-1, 1)
#     split_index = int(train_ratio * len(X_train))
#     X_val = X_train[split_index - input_length + 1:]
#     y_val = y_train[split_index - input_length + 1:]
#     X_train = X_train[:split_index]
#     y_train = y_train[:split_index]
#
#     # PyTorch数据准备
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
#     X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
#     y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     LSTMMain_model = LSTMMain(input_size=X_train.shape[2], output_size=output_length, hidden_size=dim, num_layers=num_blocks, device=device)
#     optimizer = optim.Adam(LSTMMain_model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
#
#     train_losses = []
#     val_losses = []
#
#     with tqdm.tqdm(total=num_epochs, desc=f"Training {stock_code}", unit="epoch") as pbar:
#         for epoch in range(num_epochs):
#             # 训练和验证循环
#             LSTMMain_model.train()
#             epoch_train_loss = 0
#             for inputs, targets in train_loader:
#                 outputs = LSTMMain_model(inputs)
#                 loss = criterion(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 epoch_train_loss += loss.item()
#
#             avg_train_loss = epoch_train_loss / len(train_loader)
#             train_losses.append(avg_train_loss)
#
#             LSTMMain_model.eval()
#             epoch_val_loss = 0
#             with torch.no_grad():
#                 for inputs, targets in val_loader:
#                     outputs = LSTMMain_model(inputs)
#                     val_loss = criterion(outputs, targets)
#                     epoch_val_loss += val_loss.item()
#
#             avg_val_loss = epoch_val_loss / len(val_loader)
#             val_losses.append(avg_val_loss)
#
#             pbar.set_postfix({"Train Loss": avg_train_loss, "Val Loss": avg_val_loss})
#             pbar.update(1)
#             scheduler.step()
#
#     # 使用可视化工具绘制损失曲线
#     plot_training_loss(stock_code, train_losses, val_losses, f'./data')
#
#     # 预测
#     LSTMMain_model.eval()
#     predictions = []
#     test_indices = []
#     predict_percentages = []
#     actual_percentages = []
#
#     with torch.no_grad():
#         for i in range(1 + split_index, len(features) + 1):
#             x_input = torch.tensor(features[i - input_length:i].reshape(1, input_length, -1),
#                                    dtype=torch.float32).to(device)
#             y_pred = LSTMMain_model(x_input)
#             # y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1))
#             predictions.append((1 + y_pred[0][0]) * df['close'].iloc[i - 2])
#             test_indices.append(date.index[i - 1])
#             predict_percentages.append(y_pred[0][0] * 100)
#             actual_percentages.append(labels_[i + 119])
#
#     # 使用可视化工具绘制累积收益率曲线
#     plt.figure(figsize=(14, 7))
#     plt.plot(test_indices, predict_percentages, 'g')
#     plt.plot(test_indices, actual_percentages, "r")
#     plt.legend(["forecast", "actual"], loc='upper right')
#     plt.pause(10)
#     plt.close()
#     predict_result = {str(date): pred / 100 for date, pred in zip(test_indices, predict_percentages)}
#     return predict_result, test_indices, predictions, actual_percentages

async def async_lstm_all_stock(stock_codes: List[str]):
    tasks = [lstm_stock_predict(code) for code in stock_codes]
    results = await asyncio.gather(*tasks)
