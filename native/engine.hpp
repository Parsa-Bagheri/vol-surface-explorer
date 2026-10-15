#pragma once

#include <cstdint>
#include <limits>

namespace vol {
constexpr double epsilon = 1e-8;
constexpr double min_iv = 1e-6;
constexpr double max_iv = 5.0;
constexpr double xtol = 1e-8;
constexpr int max_iterations = 200;

enum class Status : std::uint8_t {
    ok = 0, zero_volatility = 1, invalid_input = 2, outside_bounds = 3,
    unbracketed = 4, max_iterations = 5, nonfinite_result = 6
};

struct Result {
    double value = std::numeric_limits<double>::quiet_NaN();
    Status status = Status::invalid_input;
    int iterations = 0;
};

// kind is 1 for call, 0 for put; other codes are invalid contracts.
Result price(std::uint8_t kind, double spot, double strike, double time,
             double rate, double dividend, double volatility) noexcept;
Result implied_volatility(std::uint8_t kind, double spot, double strike, double time,
                          double rate, double dividend, double market_price,
                          int limit = max_iterations, double tolerance = xtol) noexcept;
}  // namespace vol
