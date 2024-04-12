import numpy as np
from typing import Callable


class CRRBinomialPricer:
    def __init__(self, S0: float, r: float, sigma: float, T: float, N: int,
                 payoff_func: Callable[[np.ndarray], np.ndarray], option_type: str = 'European'):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.payoff = payoff_func
        self.option_type = option_type
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-self.sigma * np.sqrt(self.dt))
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)

    def build_tree(self):
        stock_price_tree = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock_price_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return stock_price_tree

    def option_price(self):
        stock_price_tree = self.build_tree()
        option_price_tree = np.zeros_like(stock_price_tree)
        option_price_tree[:, self.N] = self.payoff(stock_price_tree[:, self.N])


        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = np.exp(-self.r * self.dt) * (
                        self.q * option_price_tree[j, i + 1] + (1 - self.q) * option_price_tree[j + 1, i + 1])
                if self.option_type == 'American':
                    immediate_exercise_value = self.payoff(stock_price_tree[j, i])
                    option_price_tree[j, i] = max(continuation_value, immediate_exercise_value)
                else:
                    option_price_tree[j, i] = continuation_value
        return option_price_tree[0, 0]


if __name__ == "__main__":

    def payoff(S):
        return np.abs(S - 10)
    option = CRRBinomialPricer(S0=10, r=0.05, sigma=0.20, T=2, N=4, payoff_func=payoff, option_type='American')
    print("Option Price:", option.option_price())
