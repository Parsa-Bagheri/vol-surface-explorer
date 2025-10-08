from datetime import datetime, timedelta
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_cleaner import prepare_options_data


def _black_scholes_call_price(spot: float, strike: float, time_to_expiration: float, risk_free_rate: float, volatility: float) -> float:
    if time_to_expiration <= 0 or volatility <= 0:
        raise ValueError("Time to expiration and volatility must be positive for pricing.")

    sqrt_t = np.sqrt(time_to_expiration)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)


def test_prepare_options_data_black_scholes_iv():
    spot_price = 100.0
    strike_price = 100.0
    days_to_expiration = 30
    risk_free_rate = 0.02
    true_volatility = 0.25
    time_to_expiration_years = days_to_expiration / 365.25

    theoretical_price = _black_scholes_call_price(
        spot=spot_price,
        strike=strike_price,
        time_to_expiration=time_to_expiration_years,
        risk_free_rate=risk_free_rate,
        volatility=true_volatility,
    )

    expiration_date = datetime.now().date() + timedelta(days=days_to_expiration)

    raw_df = pd.DataFrame(
        [
            {
                "strike": strike_price,
                "expirationDate": expiration_date,
                "optionType": "call",
                "impliedVolatility": 0.0,
                "volume": 50,
                "openInterest": 100,
                "bid": theoretical_price * 0.99,
                "ask": theoretical_price * 1.01,
                "lastPrice": theoretical_price,
            }
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="black-scholes",
        underlying_price=spot_price,
        risk_free_rate=risk_free_rate,
    )

    assert not cleaned_df.empty, "Cleaned DataFrame should contain the computed row"
    computed_iv = cleaned_df["impliedVolatility"].iloc[0]
    assert abs(computed_iv - true_volatility) < 1e-3, f"Expected volatility near {true_volatility}, got {computed_iv}"
    assert cleaned_df["ivComputationMethod"].iloc[0] == "black-scholes"


def test_prepare_options_data_invalid_iv_source_defaults_to_yfinance():
    spot_price = 100.0
    expiration_date = datetime.now().date() + timedelta(days=45)

    raw_df = pd.DataFrame(
        [
            {
                "strike": 105.0,
                "expirationDate": expiration_date,
                "optionType": "call",
                "impliedVolatility": 0.3,
                "volume": 10,
                "openInterest": 25,
                "bid": 5.0,
                "ask": 5.5,
                "lastPrice": 5.2,
            }
        ]
    )

    cleaned_df = prepare_options_data(
        raw_df,
        option_type_to_plot="call",
        iv_source="unknown-source",
        underlying_price=spot_price,
    )

    assert cleaned_df["ivComputationMethod"].iloc[0] == "yfinance"
    assert np.isclose(cleaned_df["impliedVolatility"].iloc[0], 0.3)
