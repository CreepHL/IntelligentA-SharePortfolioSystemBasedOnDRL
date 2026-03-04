# if __name__ == '__main__':
#     # 读取交易数据
#     df = pd.read_csv('output/results3/trade_memory_ppo_335.csv', dtype={'tic': str})
#
#     # 将日期列转换为 datetime 类型
#     df['day'] = pd.to_datetime(df['day'])
#
#     # 强制将 tic 列转换为字符串类型，保留前导零
#     df['tic'] = df['tic'].astype(str)
#
#     # 按照 tic 分组
#     grouped = df.groupby('tic')
#
#     # 设置图形大小
#     plt.figure(figsize=(14, 8))
#
#     # 定义颜色映射表，每只股票用不同的颜色
#     colors = plt.cm.get_cmap('tab10', len(grouped))
#
#     # 遍历每只股票并绘制价格走势和买卖点
#     for idx, (tic, group) in enumerate(grouped):
#         # 绘制每只股票的 low 价格走势
#         plt.plot(group['day'], group['low'], label=f'{tic}', color=colors(idx))
#
#         # 标记买入点
#         buy_points = group[group['type'] == 'buy']
#         plt.scatter(buy_points['day'], buy_points['low'], color='green', marker='^', s=50,
#                     label='Buy' if idx == 0 else "")
#
#         # 标记卖出点
#         sell_points = group[group['type'] == 'sell']
#         plt.scatter(sell_points['day'], sell_points['low'], color='red', marker='v', s=50,
#                     label='Sell' if idx == 0 else "")
#
#     # 添加标题、坐标轴标签和图例
#     plt.title('Stock Price Trends with Buy/Sell Points')
#     plt.xlabel('Date')
#     plt.ylabel('Price (Low)')
#     plt.legend(loc='upper left')
#     plt.grid(True)
#
#     # 显示图形
#     plt.tight_layout()
#     plt.show()
