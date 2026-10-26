# pragma version 0.4.3
"""
@title MockPool (crypto: price_oracle(k), packed ma_time, payable fallback)
@notice Test-only Curve pool with an EMA of block-end prices updated the way tricrypto-ng /
        stableswap-ng do it (snekmate exp): trade() sets the spot, the EMA takes the previous
        block-end spot in the first trade of a block, price_oracle decays to now.
@custom:kill Test helper, never deployed on mainnet
"""
from snekmate.utils import math

WAD: constant(uint256) = 10**18

coins: public(address[2])
ma_time: uint256
last_price: uint256
ema: uint256
last_ts: uint256


@deploy
def __init__(_coins: address[2], _p: uint256, _ma_time: uint256):
    self.coins = _coins
    self.last_price = _p
    self.ema = _p
    self.last_ts = block.timestamp
    self.ma_time = _ma_time


@internal
@view
def _ema() -> uint256:
    if self.last_ts < block.timestamp:
        alpha: uint256 = convert(math._wad_exp(-convert((block.timestamp - self.last_ts) * WAD // self.ma_time, int256)), uint256)
        return (self.last_price * (WAD - alpha) + self.ema * alpha) // WAD
    return self.ema


@external
def trade(_p: uint256):
    self.ema = self._ema()
    self.last_ts = block.timestamp
    self.last_price = _p


@external
def set_ma_time(_t: uint256):
    self.ema = self._ema()
    self.last_ts = block.timestamp
    self.ma_time = _t


@external
def set_raw(_ema: uint256, _last_price: uint256):
    self.ema = _ema
    self.last_price = _last_price
    self.last_ts = block.timestamp


@external
@view
def price_oracle(_k: uint256) -> uint256:
    return self._ema()

@external
@view
def packed_rebalancing_params() -> uint256:
    return (12345 << 128) | (678 << 64) | self.ma_time

@external
@payable
def __default__():
    pass

