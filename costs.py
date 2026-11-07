"""Fee presets, the same ones in every episode. One preset per run, printed on screen.

A cost is charged on every order: a flat commission plus a percentage of the order value
(half the bid-ask spread plus slippage). bt calls `fee(quantity, price)` for each order.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    name: str
    commission: float     # $ per order
    spread_bps: float     # half the bid-ask spread, basis points of order value
    slippage_bps: float   # price moving against you while you fill, basis points

    def fee(self, quantity, price):
        if quantity == 0:
            return 0.0
        return self.commission + abs(quantity * price) * (self.spread_bps + self.slippage_bps) / 1e4

    def short(self):
        return f"${self.commission:.2f} + {self.spread_bps + self.slippage_bps:g} bps an order"

    def describe(self):
        return (f"{self.name}: ${self.commission:.2f} an order + {self.spread_bps:g} bps spread"
                f" + {self.slippage_bps:g} bps slippage, every buy and every sell")


PRESETS = {
    "zero": Costs("No costs", 0.0, 0.0, 0.0),
    # a liquid sector ETF at a retail broker: commission-free brokers still pay the spread
    "etf_retail": Costs("Retail ETF broker", 1.00, 2.0, 3.0),
    # a stress test: wide spreads and bad fills
    "harsh": Costs("Harsh", 5.00, 10.0, 10.0),
}
