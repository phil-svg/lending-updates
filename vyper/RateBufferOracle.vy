# pragma version 0.4.3
"""
@title Rate Buffer Oracle
@author Curve.Finance
@license Copyright (c) Curve.Finance, 2020-2026 - all rights reserved
@notice Prices an exchange rate from a wrapped oracle, quoted in the wrapped
        oracle's units, and limits how fast it may rise. Each contract is
        specific to a single wrapped oracle.

        An exchange rate (e.g. a vault's share price) only grows as fast as
        the yield behind it. A jump above that pace - e.g. a bad feed round or
        a manipulated rate - would over-value collateral at once. To defend
        against this the wrapped price is buffered *asymmetrically*:

          - Upward moves are held under a buffer line that rises at most at
            `max_rise_rate`, and never sits more than `headroom` above the
            wrapped price.
          - Downward moves are passed through immediately, so a genuine loss of
            value is never hidden behind a stale, too-high price (which would
            over-value collateral - the wrong failure mode).

        The reported price is `min(p, buffer)`, where `p` is the wrapped price,
        `buffer` rises slowly (at `max_rise_rate`) and is pulled down to
        `p * (WAD + headroom) // WAD` the moment it would sit higher.
@dev `price` and `price_w` return the same value within a block (as long as the
     wrapped oracle's do), only `price_w` persists the buffer line, so the line
     advances only as often as `price_w` is called. `max_rise_rate` is per
     second and compounds on every write, as the controller's interest rate
     does, so the line never rises faster than the APY it gives. Several
     `price_w` calls in one block add nothing, as no time passes. Calls revert
     if the wrapped oracle reverts or returns a zero price.

     `max_rise_rate` and `headroom` are set by the admin of `FACTORY` (the
     DAO). The wrapped oracle is called through `price_w` before every change,
     so the old value is applied up to the change and the new one only from
     then on.
@custom:security security@curve.finance
@custom:kill There is no need to kill this contract, just kill the underlying market
"""

from curve_stablecoin.interfaces import IFactory
from curve_stablecoin.interfaces import IPriceOracle
from curve_stablecoin import constants as c

implements: IPriceOracle


event SetMaxRiseRate:
    max_rise_rate: uint256

event SetHeadroom:
    headroom: uint256


ORACLE: public(immutable(IPriceOracle))
FACTORY: public(immutable(IFactory))

WAD: constant(uint256) = c.WAD
MIN_RISE_RATE: constant(uint256) = 315522921  # 1% APY
MAX_RISE_RATE: constant(uint256) = 29055388504  # 150% APY
MIN_HEADROOM: constant(uint256) = WAD // 1000  # 0.1%
MAX_HEADROOM: constant(uint256) = WAD // 100  # 1%

max_rise_rate: public(uint256)
headroom: public(uint256)
last_buffer: public(uint256)
last_timestamp: public(uint256)


@deploy
def __init__(
    _oracle: IPriceOracle,
    _factory: IFactory,
    _max_rise_rate: uint256,
    _headroom: uint256,
):
    """
    @notice Wrap an exchange-rate oracle and seed the buffer line `_headroom`
            above its price.
    @param _oracle Wrapped price oracle, reported in the units this contract returns.
    @param _factory Factory whose `admin` (the DAO) may change `max_rise_rate`
                    and `headroom`.
    @param _max_rise_rate Fastest rise of the buffer line, per second (1e18-scaled).
    @param _headroom Most the buffer line may sit above the wrapped price (1e18 = 100%).
    """
    ORACLE = _oracle
    FACTORY = _factory

    self._set_max_rise_rate(_max_rise_rate)
    self._set_headroom(_headroom)

    # Basic sanity check for the wrapped oracle.
    p: uint256 = extcall _oracle.price_w()
    assert p > 0  # dev: invalid price

    self.last_buffer = p * (WAD + _headroom) // WAD
    self.last_timestamp = block.timestamp


@internal
@view
def _buffer(_price: uint256) -> uint256:
    """
    @notice Buffer line in the current block, given the wrapped price.
    @dev The stored line raised at `max_rise_rate` since it was written, but
         never more than `headroom` above `_price`.
    @param _price Wrapped price (1e18-scaled).
    @return The buffer line (1e18-scaled).
    """
    dt: uint256 = block.timestamp - self.last_timestamp
    risen: uint256 = self.last_buffer * (WAD + self.max_rise_rate * dt) // WAD
    return min(risen, _price * (WAD + self.headroom) // WAD)


@internal
def _price_w() -> uint256:
    """
    @notice Buffered wrapped price, persisting the buffer line.
    """
    p: uint256 = extcall ORACLE.price_w()
    assert p > 0  # dev: invalid price
    buffer: uint256 = self._buffer(p)

    self.last_buffer = buffer
    self.last_timestamp = block.timestamp

    return min(p, buffer)


@internal
def _set_max_rise_rate(_max_rise_rate: uint256):
    """
    @notice Validates and stores how fast the buffer line may rise.
    @dev Shared by the constructor and set_max_rise_rate; performs no access control.
    @param _max_rise_rate Fastest rise of the buffer line, per second (1e18-scaled).
    """
    assert _max_rise_rate >= MIN_RISE_RATE, "max_rise_rate too low"
    assert _max_rise_rate <= MAX_RISE_RATE, "max_rise_rate too high"

    self.max_rise_rate = _max_rise_rate

    log SetMaxRiseRate(max_rise_rate=_max_rise_rate)


@internal
def _set_headroom(_headroom: uint256):
    """
    @notice Validates and stores the headroom of the buffer line.
    @dev Shared by the constructor and set_headroom; performs no access control.
    @param _headroom Most the buffer line may sit above the wrapped price (1e18 = 100%).
    """
    assert _headroom >= MIN_HEADROOM, "headroom too low"
    assert _headroom <= MAX_HEADROOM, "headroom too high"

    self.headroom = _headroom

    log SetHeadroom(headroom=_headroom)


@external
@view
def price() -> uint256:
    """
    @notice Wrapped price, held under the slowly rising buffer line.
    @return The buffered price, 1e18-scaled and quoted in the wrapped oracle's
            units.
    """
    p: uint256 = staticcall ORACLE.price()
    assert p > 0  # dev: invalid price
    return min(p, self._buffer(p))


@external
def price_w() -> uint256:
    """
    @notice Same as `price`, but persists the buffer line.
    @return The buffered price, 1e18-scaled and quoted in the wrapped oracle's
            units.
    """
    return self._price_w()


@external
def set_max_rise_rate(_max_rise_rate: uint256):
    """
    @notice Admin function to change how fast the buffer line may rise.
    @dev The buffer line is written with the old rate first, so a change never
         applies backwards.
    @param _max_rise_rate Fastest rise of the buffer line, per second (1e18-scaled).
    """
    assert msg.sender == staticcall FACTORY.admin(), "Not factory admin"

    self._price_w()
    self._set_max_rise_rate(_max_rise_rate)


@external
def set_headroom(_headroom: uint256):
    """
    @notice Admin function to change the headroom of the buffer line.
    @dev The buffer line is written with the old headroom first, so a change never
         applies backwards. A higher headroom lifts the line only at
         `max_rise_rate`, never at once.
    @param _headroom Most the buffer line may sit above the wrapped price (1e18 = 100%).
    """
    assert msg.sender == staticcall FACTORY.admin(), "Not factory admin"

    self._price_w()
    self._set_headroom(_headroom)
