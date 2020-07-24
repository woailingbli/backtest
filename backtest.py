import numpy as np
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt
import datetime
import dateutil
import pandas_datareader as web
import math

trade_cal=pd.read_csv('trade_cal')

class G:
    pass

class Context:
    def __init__(self,cash,start_date,end_date):
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {}
        self.benchmark = None
        self.date_range = trade_cal[(trade_cal['is_open'] == 1) &\
                                    (trade_cal['cal_date'] >= start_date) &\
                                    (trade_cal['cal_date'] <= end_date)]['cal_date']
        self.dt = None


def set_benchmark(security): #设置基准
    context.benchmark = security

def get_data_from_yahoo(security,start_date,end_date):#从网络下载回测数据
    df = web.DataReader(security,'yahoo',start_date, end_date )
    return df

def attribute_daterange_history(security,start_date,end_date,fields=('Open','Close','High','Low','Volume')):
    filename = security+'.csv'
    try:
        f = open(filename,'r')
        df = pd.read_csv(security+'.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index(['Date'], inplace=True)
        df=df[start_date:end_date]
    except FileNotFoundError:
        print('download data')
        df = get_data_from_yahoo(security,start_date,end_date)
        print('finish')
    return df[list(fields)]

def attribute_history(security,count,fields=('Open','Close','High','Low','Volume','Adj Close')):#读取回测数据
    end_date = (context.dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = trade_cal[(trade_cal['is_open']==1) & (trade_cal['cal_date']<=end_date)][-count:].iloc[0,:]['cal_date']#向前取n个交易日
    return attribute_daterange_history(security,start_date,end_date)


def get_today_data(security):
    today = context.dt.strftime('%Y-%m-%d')
    filename = security+'.csv'
    try:
        f = open(filename,'r')
        df = pd.read_csv(f)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index(['Date'], inplace=True)
        data = df.loc[today,:]
    except FileNotFoundError:
        print('download data')
        data = get_data_from_yahoo(security,'yahoo', today,today).iloc[0,:]
        print('finish')
    except KeyError:
        data = pd.Series(dtype=float)
    return data

def _order(today_data,security,quantity):#下单函数，ToDO:处理停牌,涨跌停
    p = today_data['Open']
    if quantity >= 0:#大于0表示买入
        if context.cash - quantity * p < 0:
            quantity = int(context.cash / p)
            print("现金不足,下单数已调整为%d" % (quantity))
        if quantity % 100 != 0 :
            quantity = int(quantity / 100)*100
            print("调整下单为100的倍数，%d" % (quantity))
    else:#小于0表示卖出
        if quantity % 100 != 0:
            if quantity != -context.positions.get(security,0):
                quantity = int(quantity / 100) * 100
                print("调整下单为100的倍数，%d" % (quantity))

        if context.positions.get(security,0) < -quantity:
            quantity = -context.positions.get(security,0)
            print("卖出股票不能超过持仓，已经调整为%d" % quantity)

    context.positions[security] = context.positions.get(security,0) + quantity
    context.cash = context.cash - quantity * p

    if context.positions[security] == 0:
        del context.positions[security]

def order(security,quantity):
    today_data = get_today_data(security)
    _order(today_data,security,quantity)

def order_target(security,target_quantity):
    if target_quantity < 0:
        print("持仓最低为0，已调整为0")
        target_quantity = 0

    today_data = get_today_data(security)
    hold_quantity = context.positions.get(security,0)
    quantity = target_quantity - hold_quantity
    _order(today_data,security,quantity)

def order_value(security, value):
    today_data = get_today_data(security)
    quantity = int(value / today_data['Open'])
    _order(today_data,security,quantity)

def order_target_value(security,target_value):
    if target_value < 0.0:
        print("持仓最低为0，已调整为0")
        target_value = 0.0

    today_data = get_today_data(security)
    hold_value = context.positions.get(security,0) * today_data['Open']
    quantity = int((target_value - hold_value)/today_data['Open'])
    _order(today_data,security,quantity)
#
# def plot_result(backtest_df):
#     init_value = context.cash
#     backtest_df['total_portfolio_return'] = (backtest_df['portfolio_value'] - init_value)/init_value
#     backtest_df['daily_portfolio_return'] = (backtest_df['portfolio_value'] - backtest_df['portfolio_value'].shift(1)) \
#                                             /backtest_df['portfolio_value'].shift(1)
#     backtest_df['daily_value_change'] = backtest_df['portfolio_value'] - backtest_df['portfolio_value'].shift(1)
#
#     bm_df = attribute_daterange_history(context.benchmark,context.start_date,context.end_date)
#     bm_init = bm_df['Open'][0]
#     backtest_df['total_benchmark_return'] = (bm_df['Open'] - bm_init) / bm_init
#     backtest_df['benchmark_value'] = init_value * (1+backtest_df['total_benchmark_return'])
#     backtest_df['daily_benchmark_return'] = (bm_df['Open'] - bm_df['Open'].shift(1)) / bm_df['Open'].shift(1)
#
#     backtest_df = backtest_df.dropna()
#     annual_benchmark_return = (math.pow((bm_df['Open'][-1] / bm_init) , (365/len(bm_df))) - 1)
#
#     annual_portfilio_return = (math.pow(backtest_df['portfolio_value'][-1] / backtest_df['portfolio_value'][0],(365/len(backtest_df))) - 1)
#
#     covxy = np.cov(backtest_df['daily_benchmark_return'][1:].values.astype(float),
#                    backtest_df['daily_portfolio_return'][1:].values.astype(float))
#     beta = covxy[0][1]/np.var(backtest_df['daily_benchmark_return'][1:].values)
#
#     annual_portfilio_volatility = np.std(backtest_df['daily_portfolio_return'][1:].values) * math.sqrt(252)
#
#     alpha = (annual_portfilio_return - g.Rf) - beta * (annual_benchmark_return-g.Rf)
#
#     sharpe = (annual_portfilio_return - g.Rf) / annual_portfilio_volatility
#
#     tracking_error = np.std(backtest_df['daily_portfolio_return'][1:].values.astype(float)
#                             - backtest_df['daily_benchmark_return'][1:].values.astype(float)) \
#                      * math.sqrt(252)
#
#     IR = (annual_portfilio_return - annual_benchmark_return)/tracking_error
#
#     backtest_df['max_total_value'] = backtest_df['portfolio_value'].expanding().max()
#     backtest_df['drawdown'] = backtest_df['portfolio_value'] / backtest_df['max_total_value']
#     max_drawdown_end = backtest_df.sort_values(by=['drawdown']).iloc[[0], :]
#     max_drawdown_start = backtest_df[backtest_df.index <= max_drawdown_end.index[0]].sort_values \
#         (by=['portfolio_value'], ascending=False).iloc[[0], :]
#
#     print('基准年化收益：%5.2f' % (100 * annual_benchmark_return))
#     print('策略年化收益：%5.2f' % (100 * annual_portfilio_return))
#     print('beta: %5.2f' % beta)
#     print('alpha: %5.2f' % alpha)
#     print(('年化收益率波动率： %5.2f' % annual_portfilio_volatility))
#     print(('夏普比率： %5.2f' % sharpe))
#     print(('信息比率： %5.2f' % IR))
#     print('最大回撤%5.2f%%' % ((1 - max_drawdown_end['drawdown'].values) * 100))
#     print('最大回撤区间为: [%s' % max_drawdown_start.index[0] + ',%s]' % max_drawdown_end.index[0])
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(2,1,1)
#     ax1.plot(backtest_df['total_portfolio_return'],color='blue',label='portfolio_return')
#     ax1.plot(backtest_df['total_benchmark_return'],color='red',label='benchmark_return')
#     ax1.plot(max_drawdown_start['total_portfolio_return'],marker='o',color='green')
#     ax1.plot(max_drawdown_end['total_portfolio_return'],marker='o',color='green')
#     plt.legend()
#     plt.xticks(rotation=90)
#     plt.ylabel('Return')
#
#     ax2 = fig.add_subplot(2,1,2)
#     ax2.bar(backtest_df['daily_value_change'].index,backtest_df['daily_value_change'].values)
#     plt.ylabel('Daily_Change')
#
#     plt.xticks(rotation=45)
#     plt.subplots_adjust(top=0.88,
#                         bottom=0.11,
#                         left=0.17,
#                         right=0.9,
#                         hspace=0.0,
#                         wspace=0.2)
#     plt.show()

def run():
    backtest_df = pd.DataFrame(index=pd.to_datetime(context.date_range),columns=['portfolio_value'])
    init_value = context.cash
    initialize(context)
    last_price = {}
    for dt in context.date_range:
        context.dt = dateutil.parser.parse(dt)
        handle_data(context)
        value = context.cash
        for stock in context.positions:
            today_data = get_today_data(stock)
            if len(today_data) == 0:
                p = last_price[stock]
                value = value + context.positions[stock] * p
                position = context.positions[stock]
            else:
                p = today_data['Open']
                value = value + context.positions[stock] * p
                last_price[stock] = p
                position = context.positions[stock]
        backtest_df.loc[dt,'portfolio_value'] = value

    backtest_df['total_portfolio_return'] = (backtest_df['portfolio_value'] - init_value)/init_value
    backtest_df['daily_portfolio_return'] = (backtest_df['portfolio_value'] - backtest_df['portfolio_value'].shift(1)) \
                                            /backtest_df['portfolio_value'].shift(1)
    backtest_df['daily_value_change'] = backtest_df['portfolio_value'] - backtest_df['portfolio_value'].shift(1)

    bm_df = attribute_daterange_history(context.benchmark,context.start_date,context.end_date)
    bm_init = bm_df['Open'][0]
    backtest_df['total_benchmark_return'] = (bm_df['Open'] - bm_init) / bm_init
    backtest_df['benchmark_value'] = init_value * (1+backtest_df['total_benchmark_return'])
    backtest_df['daily_benchmark_return'] = (bm_df['Open'] - bm_df['Open'].shift(1)) / bm_df['Open'].shift(1)

    backtest_df = backtest_df.dropna()
    #计算基准年化回报率
    annual_benchmark_return = (math.pow((bm_df['Open'][-1] / bm_init) , (365/len(bm_df))) - 1)
    #计算组合年化回报率
    annual_portfilio_return = (math.pow(backtest_df['portfolio_value'][-1] / backtest_df['portfolio_value'][0],(365/len(backtest_df))) - 1)
    #计算beta
    covxy = np.cov(backtest_df['daily_benchmark_return'][1:].values.astype(float),
                   backtest_df['daily_portfolio_return'][1:].values.astype(float))
    beta = covxy[0][1]/np.var(backtest_df['daily_benchmark_return'][1:].values)
    #计算年化收益率波动率
    annual_portfilio_volatility = np.std(backtest_df['daily_portfolio_return'][1:].values) * math.sqrt(252)
    #计算alpha
    alpha = (annual_portfilio_return - g.Rf) - beta * (annual_benchmark_return-g.Rf)
    #计算夏普比率
    sharpe = (annual_portfilio_return - g.Rf) / annual_portfilio_volatility
    #计算IR
    tracking_error = np.std(backtest_df['daily_portfolio_return'][1:].values.astype(float)
                            - backtest_df['daily_benchmark_return'][1:].values.astype(float)) \
                     * math.sqrt(252)
    IR = (annual_portfilio_return - annual_benchmark_return)/tracking_error
    #计算最大回撤及回撤区间
    backtest_df['max_total_value'] = backtest_df['portfolio_value'].expanding().max()
    backtest_df['drawdown'] = backtest_df['portfolio_value'] / backtest_df['max_total_value']
    max_drawdown_end = backtest_df.sort_values(by=['drawdown']).iloc[[0], :]
    max_drawdown_start = backtest_df[backtest_df.index <= max_drawdown_end.index[0]].sort_values \
        (by=['portfolio_value'], ascending=False).iloc[[0], :]

    print('基准年化收益：%5.2f' % (100 * annual_benchmark_return))
    print('策略年化收益：%5.2f' % (100 * annual_portfilio_return))
    print('beta: %5.2f' % beta)
    print('alpha: %5.2f' % alpha)
    print(('年化收益率波动率： %5.2f' % annual_portfilio_volatility))
    print(('夏普比率： %5.2f' % sharpe))
    print(('信息比率： %5.2f' % IR))
    print('最大回撤%5.2f%%' % ((1 - max_drawdown_end['drawdown'].values) * 100))
    print('最大回撤区间为: [%s' % max_drawdown_start.index[0] + ' , %s]' % max_drawdown_end.index[0])

    #画图
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(backtest_df['total_portfolio_return'],color='blue',label='portfolio_return')
    ax1.plot(backtest_df['total_benchmark_return'],color='red',label='benchmark_return')
    ax1.plot(max_drawdown_start['total_portfolio_return'],marker='o',color='green')
    ax1.plot(max_drawdown_end['total_portfolio_return'],marker='o',color='green')
    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel('Return')

    ax2 = fig.add_subplot(2,1,2)
    ax2.bar(backtest_df['daily_value_change'].index,backtest_df['daily_value_change'].values)
    plt.ylabel('Daily_Change')

    plt.xticks(rotation=45)
    plt.subplots_adjust(top=0.88,
                        bottom=0.11,
                        left=0.17,
                        right=0.9,
                        hspace=0.0,
                        wspace=0.2)

    plt.title('result')
    plt.show()


Cash = 1000000
START_DATE = '2013-01-01'
END_DATE = '2019-01-01'

g=G()
context = Context(Cash,START_DATE,END_DATE)

def initialize(context):#设置全局变量
    set_benchmark('000001.SZ')
    g.security = '000001.SZ'
    g.Rf = 0.01

def handle_data(context):#编写策略
    hist = attribute_history(g.security,60)
    ma20 = hist['Close'][-20:].mean()
    ma60 = hist['Close'].mean()

    if (ma20 > ma60) and (g.security not in context.positions):
        print('%s，买入股票' % context.dt)
        order_value(g.security,context.cash)

    elif (ma20 < ma60 ) and (g.security in context.positions):
        print('%s，卖出股票' % context.dt)
        order_target_value(g.security,0)

run()