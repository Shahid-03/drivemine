#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

using darr = py::array_t<
    double,
    py::array::c_style | py::array::forcecast
>;

py::array_t<long> hard_braking(
    darr x,
    darr y,
    darr ts,
    double threshold = 3.0
) {
    if (x.ndim() != 1 ||
        y.ndim() != 1 ||
        ts.ndim() != 1 ||
        x.size() != y.size() ||
        x.size() != ts.size()) {
        throw std::runtime_error(
            "x, y, ts must be 1D arrays with equal length"
        );
    }

    const ssize_t n = x.size();

    if (n < 3) {
        return py::array_t<long>(0);
    }

    const double* px = x.data();
    const double* py_data = y.data();
    const double* pt = ts.data();

    std::vector<long> result;
    result.reserve(n / 10);

    {
        py::gil_scoped_release release;

        double dt1 = (pt[1] - pt[0]) * 1e-6;

        if (dt1 <= 0.0) {
            throw std::runtime_error(
                "timestamps must be strictly increasing"
            );
        }

        double dx1 = px[1] - px[0];
        double dy1 = py_data[1] - py_data[0];

        double v1 = std::hypot(dx1, dy1) / dt1;

        for (ssize_t i = 2; i < n; ++i) {
            double dt2 = (pt[i] - pt[i - 1]) * 1e-6;

            if (dt2 <= 0.0) {
                throw std::runtime_error(
                    "timestamps must be strictly increasing"
                );
            }

            double dx = px[i] - px[i - 1];
            double dy = py_data[i] - py_data[i - 1];

            double v2 = std::hypot(dx, dy) / dt2;

            // Avoid another division:
            // (v2 - v1) / dt2 < -threshold
            // is equivalent to:
            // (v2 - v1) < -threshold * dt2
            if ((v2 - v1) < -threshold * dt2) {
                result.push_back(i);
            }

            v1 = v2;
        }
    }

    py::array_t<long> output(result.size());
    auto out = output.mutable_unchecked<1>();

    for (ssize_t i = 0; i < static_cast<ssize_t>(result.size()); ++i) {
        out(i) = result[i];
    }

    return output;
}


PYBIND11_MODULE(cpp_miners, m) {
    m.doc() = "C++ scenario mining kernels";

    m.def(
        "hard_braking",
        &hard_braking,
        py::arg("x"),
        py::arg("y"),
        py::arg("ts"),
        py::arg("threshold") = 3.0
    );
}
