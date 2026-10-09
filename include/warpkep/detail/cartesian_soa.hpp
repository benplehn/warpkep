#pragma once

namespace warpkep::detail {


template <typename T>
struct CartesianSoAConstView {
    const T* r_x;
    const T* r_y;
    const T* r_z;

    const T* v_x;
    const T* v_y;
    const T* v_z;
};


template <typename T>
struct CartesianSoAView {
    T* r_x;
    T* r_y;
    T* r_z;

    T* v_x;
    T* v_y;
    T* v_z;
};

} // namespace warpkep::detail