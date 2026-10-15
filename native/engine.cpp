#include "engine.hpp"

#include <algorithm>
#include <cmath>

namespace vol {
namespace {
const double inv_sqrt_two = 1.0 / std::sqrt(2.0);  // Match the scalar Python reference.

struct Terms {
    bool call;
    double log_moneyness, discounted_spot, discounted_strike, time, sqrt_time, carry;
    double lower() const noexcept {
        return std::max(0.0, call ? discounted_spot - discounted_strike
                                  : discounted_strike - discounted_spot);
    }
    double upper() const noexcept { return call ? discounted_spot : discounted_strike; }
    double evaluate(double sigma) const noexcept {
        if (sigma == 0.0) return lower();
        const double width = sigma * sqrt_time;
        const double d1 = (log_moneyness + (carry + 0.5 * sigma * sigma) * time) / width;
        const double d2 = d1 - width;
        const auto cdf = [](double x) { return 0.5 * std::erfc(-x * inv_sqrt_two); };
        return call ? discounted_spot * cdf(d1) - discounted_strike * cdf(d2)
                    : discounted_strike * cdf(-d2) - discounted_spot * cdf(-d1);
    }
};

bool valid(std::uint8_t kind, double spot, double strike, double time,
           double rate, double dividend, double value) noexcept {
    return kind <= 1 && std::isfinite(spot) && spot > 0 && std::isfinite(strike) && strike > 0
        && std::isfinite(time) && time > 0 && std::isfinite(rate) && std::isfinite(dividend)
        && std::isfinite(value) && value >= 0;
}

Terms terms(std::uint8_t kind, double spot, double strike, double time,
            double rate, double dividend) noexcept {
    return {kind == 1, std::log(spot / strike), spot * std::exp(-dividend * time),
            strike * std::exp(-rate * time), time, std::sqrt(time), rate - dividend};
}

bool finite(const Terms& t) noexcept {
    return std::isfinite(t.log_moneyness) && std::isfinite(t.discounted_spot)
        && std::isfinite(t.discounted_strike);
}

Result failure(Status status, int iterations = 0) noexcept {
    return {std::numeric_limits<double>::quiet_NaN(), status, iterations};
}

// Brent-Dekker: preserve a sign-changing bracket, interpolate only when the
// proposed step is bounded and improving, otherwise bisect. No derivative or
// unbounded Newton step can leave [min_iv, max_iv]. Terms are reused per root.
Result solve(const Terms& t, double target, int limit, double tolerance) noexcept {
    double previous = min_iv, current = max_iv, bracket = min_iv;
    double f_previous = t.evaluate(previous) - target;
    double f_current = t.evaluate(current) - target;
    double f_bracket = f_previous;
    if (!std::isfinite(f_previous) || !std::isfinite(f_current))
        return failure(Status::nonfinite_result);
    if (f_previous == 0) return {previous, Status::ok, 0};
    if (f_current == 0) return {current, Status::ok, 0};
    if (std::signbit(f_previous) == std::signbit(f_current)) return failure(Status::unbracketed);
    double last_step = current - previous, step = last_step;
    for (int iteration = 1; iteration <= limit; ++iteration) {
        if (f_previous != 0 && std::signbit(f_previous) != std::signbit(f_current)) {
            bracket = previous;
            f_bracket = f_previous;
            last_step = step = current - previous;
        }
        if (std::abs(f_bracket) < std::abs(f_current)) {
            previous = current;
            current = bracket;
            bracket = previous;
            f_previous = f_current;
            f_current = f_bracket;
            f_bracket = f_previous;
        }
        const double delta = (tolerance + 4 * std::numeric_limits<double>::epsilon() * std::abs(current)) / 2;
        const double bisect = (bracket - current) / 2;
        if (f_current == 0 || std::abs(bisect) < delta) return {current, Status::ok, iteration};
        if (std::abs(last_step) > delta && std::abs(f_current) < std::abs(f_previous)) {
            double trial;
            if (previous == bracket) {
                trial = -f_current * (current - previous) / (f_current - f_previous);
            } else {
                const double slope_previous = (f_previous - f_current) / (previous - current);
                const double slope_bracket = (f_bracket - f_current) / (bracket - current);
                trial = -f_current * (f_bracket * slope_bracket - f_previous * slope_previous)
                    / (slope_bracket * slope_previous * (f_bracket - f_previous));
            }
            if (std::isfinite(trial) && 2 * std::abs(trial) < std::min(std::abs(last_step), 3 * std::abs(bisect) - delta)) {
                last_step = step;
                step = trial;
            } else {
                last_step = step = bisect;
            }
        } else {
            last_step = step = bisect;
        }
        previous = current;
        f_previous = f_current;
        current += std::abs(step) > delta ? step : std::copysign(delta, bisect);
        f_current = t.evaluate(current) - target;
        if (!std::isfinite(f_current)) return failure(Status::nonfinite_result, iteration);
    }
    return failure(Status::max_iterations, limit);
}
}  // namespace

Result price(std::uint8_t kind, double spot, double strike, double time,
             double rate, double dividend, double volatility) noexcept {
    if (!valid(kind, spot, strike, time, rate, dividend, volatility)) return {};
    const auto t = terms(kind, spot, strike, time, rate, dividend);
    if (!finite(t)) return failure(Status::nonfinite_result);
    const double value = t.evaluate(volatility);
    return std::isfinite(value) ? Result{value, volatility == 0 ? Status::zero_volatility : Status::ok, 0}
                                : failure(Status::nonfinite_result);
}

Result implied_volatility(std::uint8_t kind, double spot, double strike, double time,
                          double rate, double dividend, double market_price,
                          int limit, double tolerance) noexcept {
    if (!valid(kind, spot, strike, time, rate, dividend, market_price)
        || limit < 0 || !std::isfinite(tolerance) || tolerance <= 0) return {};
    const auto t = terms(kind, spot, strike, time, rate, dividend);
    if (!finite(t)) return failure(Status::nonfinite_result);
    if (std::abs(market_price - t.lower()) <= epsilon) return {0, Status::zero_volatility, 0};
    if (market_price < t.lower() - epsilon || market_price > t.upper() + epsilon)
        return failure(Status::outside_bounds);
    return solve(t, market_price, limit, tolerance);
}
}  // namespace vol
