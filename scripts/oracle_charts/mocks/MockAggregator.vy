# pragma version 0.4.3
"""
@title MockAggregator
@notice Test-only crvUSD stable aggregator: price set by the test, price_w() counts its calls.
        last_price / last_timestamp follow the writes as in AggregateStablePrice2: price_w()
        stores the price and the block timestamp once per block, set_price() changes only
        what price() returns until the next price_w().
@custom:kill Test helper, never deployed on mainnet
"""

stablecoin: public(address)
p: uint256
n_price_w: public(uint256)
last_price: public(uint256)
last_timestamp: public(uint256)


@deploy
def __init__(_stablecoin: address, _p: uint256):
    self.stablecoin = _stablecoin
    self.p = _p
    self.last_price = _p
    self.last_timestamp = block.timestamp


@external
def set_price(_p: uint256):
    self.p = _p


@external
@view
def price() -> uint256:
    return self.p


@external
def price_w() -> uint256:
    self.n_price_w += 1
    if self.last_timestamp == block.timestamp:
        return self.last_price
    self.last_timestamp = block.timestamp
    self.last_price = self.p
    return self.p
