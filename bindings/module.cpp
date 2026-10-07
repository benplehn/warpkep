#include <nanobind/nanobind.h>
#include <warpkep/version.hpp>

NB_MODULE(_core, m) {
    m.def("version", &warpkep::version);
}