#include "engine.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

void check(bool condition, const char* message) {
    // Do not use assert(): Release builds define NDEBUG.
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

int main() {
    using vol::Status;
    for (std::uint8_t kind : {0, 1}) {
        for (double time : {1e-5, 0.1, 1.0}) {
            for (double sigma : {0.0, 0.2, 1.5, 5.0}) {
                const auto p = vol::price(kind, 100, 100, time, .03, .01, sigma);
                check(std::isfinite(p.value), "finite BSM price");
                const auto iv = vol::implied_volatility(kind, 100, 100, time, .03, .01, p.value);
                check(std::abs(iv.value - sigma) <= 1e-8, "round trip IV including boundaries");
            }
        }
    }
    const auto call = vol::price(1, 100, 105, .5, .03, .01, .24);
    const auto put = vol::price(0, 100, 105, .5, .03, .01, .24);
    check(std::abs(call.value - put.value - (100 * std::exp(-.005) - 105 * std::exp(-.015))) < 1e-12, "put call parity");
    check(vol::price(2, 100, 100, 1, 0, 0, .2).status == Status::invalid_input, "invalid type");
    check(vol::price(1, 100, 0, 1, 0, 0, .2).status == Status::invalid_input, "invalid strike");
    check(vol::price(1, 100, 100, 0, 0, 0, .2).status == Status::invalid_input, "expired option");
    check(vol::price(1, 100, 100, 1, 0, 0, -1).status == Status::invalid_input, "negative volatility");
    check(vol::price(1, 100, 100, 1, 0, 0, std::numeric_limits<double>::infinity()).status == Status::invalid_input, "nonfinite input");
    check(vol::price(1, 100, 100, 1, -1000, 0, .2).status == Status::nonfinite_result, "overflow");
    check(vol::implied_volatility(1, 100, 100, 1, 0, 0, 101).status == Status::outside_bounds, "model bounds");
    check(vol::implied_volatility(1, 100, 100, 1e-5, 0, 0, 99).status == Status::unbracketed, "root beyond max IV");
    check(vol::implied_volatility(1, 100, 100, 1, 0, 0, 10, 1).status == Status::max_iterations, "failed convergence");
    check(vol::implied_volatility(1, 100, 90, 1, 0, 0, 10 + 0.5e-8).status == Status::zero_volatility, "lower bound epsilon");
    std::cout << "Native pricing, inversion, boundary and failure tests passed\n";
}
