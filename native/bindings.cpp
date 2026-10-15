#include "engine.hpp"
#include "build_info.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

template <typename T>
std::vector<T> owned_input(const py::array& array, py::ssize_t length, const char* name) {
    if (!array.dtype().is(py::dtype::of<T>()))
        throw py::type_error(std::string(name) + " has an incorrect dtype");
    if (array.ndim() != 1 || array.size() != length)
        throw py::value_error(std::string(name) + " must be one dimensional and match contracts length");
    if (!(array.flags() & py::array::c_style))
        throw py::value_error(std::string(name) + " must be C contiguous");
    // Copy while holding the GIL. The computation never races with a Python
    // thread mutating a borrowed input and never retains a caller's pointer.
    std::vector<T> values(static_cast<std::size_t>(length));
    if (length) std::memcpy(values.data(), array.data(), values.size() * sizeof(T));
    return values;
}

py::tuple batch(const py::array& kinds, const py::array& spots, const py::array& strikes,
                const py::array& times, const py::array& rates, const py::array& dividends,
                const py::array& inputs, bool invert, int limit, double tolerance) {
    if (limit < 0 || !std::isfinite(tolerance) || tolerance <= 0)
        throw py::value_error("maxiter must be nonnegative and xtol must be finite and positive");
    const auto n = kinds.size();
    const auto k = owned_input<std::uint8_t>(kinds, n, "kinds");
    const auto s = owned_input<double>(spots, n, "spots");
    const auto x = owned_input<double>(strikes, n, "strikes");
    const auto t = owned_input<double>(times, n, "times");
    const auto r = owned_input<double>(rates, n, "rates");
    const auto q = owned_input<double>(dividends, n, "dividends");
    const auto v = owned_input<double>(inputs, n, "inputs");
    py::array_t<double> values(n);
    py::array_t<std::uint8_t> statuses(n);
    py::array_t<std::int32_t> iterations(n);
    auto* out = values.mutable_data();
    auto* status = statuses.mutable_data();
    auto* counts = iterations.mutable_data();
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            const auto result = invert ? vol::implied_volatility(k[i], s[i], x[i], t[i], r[i], q[i], v[i], limit, tolerance)
                                       : vol::price(k[i], s[i], x[i], t[i], r[i], q[i], v[i]);
            out[i] = result.value;
            status[i] = static_cast<std::uint8_t>(result.status);
            counts[i] = result.iterations;
        }
    }
    return py::make_tuple(values, statuses, iterations);
}

PYBIND11_MODULE(_vol_native, m) {
    m.doc() = "Batched, single-threaded BSM engine; owns inputs while the GIL is released";
    m.def("batch", &batch, py::arg("kinds"), py::arg("spots"), py::arg("strikes"),
          py::arg("times"), py::arg("rates"), py::arg("dividends"), py::arg("inputs"),
          py::arg("invert"), py::arg("maxiter") = vol::max_iterations, py::arg("xtol") = vol::xtol);
    m.attr("source_sha256") = VOL_SOURCE_SHA;
    m.attr("compiler") = VOL_COMPILER;
    m.attr("build_type") = VOL_BUILD_TYPE;
    m.attr("engine_version") = "bsm-brent-v1";
}
