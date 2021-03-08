#!/usr/bin/python

# The MIT License (MIT)
#
# Copyright (c) 2015 Christian Zielinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULtAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import portfolioopt as pfopt


def section(caption):
    print('\n\n' + str(caption))
    print('-' * len(caption))


def print_portfolio_info(returns, avg_rets, weights):
    """
    Print information on expected portfolio performance.
    """
    ret = (weights * avg_rets).sum()
    std = (weights * returns).sum(1).std()
    sharpe = ret / std
    print("Optimal weights:\n{}\n".format(weights))
    print("Expected return:   {}".format(ret))
    print("Expected variance: {}".format(std**2))
    print("Expected Sharpe:   {}".format(sharpe))

def load_data():
    date_name = 'trade_date' # trade_date or Date
    price_name = 'pct_chg'  # pct_chg or pct_change
    re = np.zeros((178, 5))
    cur_stock = pd.read_csv('./data/1.csv', encoding='gbk', parse_dates=[date_name])
    cur_stock = cur_stock[cur_stock[date_name] >= pd.to_datetime('2020-03-01')]
    cur_stock = cur_stock.reset_index(drop=True)  # 重排序
    re[:,0] = cur_stock[price_name].to_numpy()
    cur_stock = pd.read_csv('./data/2.csv', encoding='gbk', parse_dates=[date_name])
    cur_stock = cur_stock[cur_stock[date_name] >= pd.to_datetime('2020-03-01')]
    cur_stock = cur_stock.reset_index(drop=True)
    re[:,1] = cur_stock[price_name].to_numpy()
    cur_stock = pd.read_csv('./data/3.csv', encoding='gbk', parse_dates=[date_name])
    cur_stock = cur_stock[cur_stock[date_name] >= pd.to_datetime('2020-03-01')]
    cur_stock = cur_stock.reset_index(drop=True)
    re[:,2] = cur_stock[price_name].to_numpy()
    cur_stock = pd.read_csv('./data/4.csv', encoding='gbk', parse_dates=[date_name])
    cur_stock = cur_stock[cur_stock[date_name] >= pd.to_datetime('2020-03-01')]
    cur_stock = cur_stock.reset_index(drop=True)
    re[:,3] = cur_stock[price_name].to_numpy()
    cur_stock = pd.read_csv('./data/5.csv', encoding='gbk', parse_dates=[date_name])
    cur_stock = cur_stock[cur_stock[date_name] >= pd.to_datetime('2020-03-01')]
    cur_stock = cur_stock.reset_index(drop=True)
    re[:,4] = cur_stock[price_name].to_numpy()

    dates = cur_stock[date_name]
    assets = ['asset_a', 'asset_b', 'asset_c', 'asset_d', 'asset_e']
    returns = pd.DataFrame(re, columns=assets, index=dates)
    avg_rets = returns.mean()
    cov_mat = returns.cov()  
    return returns, cov_mat, avg_rets

def main():

    # returns, cov_mat, avg_rets = pfopt.create_test_data(num_days=1000)
    returns, cov_mat, avg_rets = load_data()
    # yy = returns['asset_a']
    # xx = [ii for ii in range(len(yy))]
    # plt.plot(xx, yy)
    # plt.hlines(avg_rets[0], xmin=0, xmax=100, colors='black')
    # plt.show()

    # return 0

    section("Example returns")
    print(returns.head(5))
    print("...")

    section("Average returns")
    print(avg_rets)

    section("Covariance matrix")
    print(cov_mat)

    section("Minimum variance portfolio (long only)")
    weights = pfopt.min_var_portfolio(cov_mat)
    print_portfolio_info(returns, avg_rets, weights)

    section("Minimum variance portfolio (long/short)")
    weights = pfopt.min_var_portfolio(cov_mat, allow_short=True)
    print_portfolio_info(returns, avg_rets, weights)

    # Define some target return, here the 70% quantile of the average returns
    target_ret = avg_rets.quantile(0.7)

    section("Markowitz portfolio (long only, target return: {:.5f})".format(target_ret))
    weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target_ret)
    print_portfolio_info(returns, avg_rets, weights)

    section("Markowitz portfolio (long/short, target return: {:.5f})".format(target_ret))
    weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target_ret, allow_short=True)
    print_portfolio_info(returns, avg_rets, weights)

    section("Markowitz portfolio (market neutral, target return: {:.5f})".format(target_ret))
    weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target_ret, allow_short=True,
                                                                       market_neutral=True)
    print_portfolio_info(returns, avg_rets, weights)

    section("Tangency portfolio (long only)")
    weights = pfopt.tangency_portfolio(cov_mat, avg_rets)
    weights = pfopt.truncate_weights(weights)   # Truncate some tiny weights
    print_portfolio_info(returns, avg_rets, weights)

    section("Tangency portfolio (long/short)")
    weights = pfopt.tangency_portfolio(cov_mat, avg_rets, allow_short=True)
    print_portfolio_info(returns, avg_rets, weights)


if __name__ == '__main__':
    main()
