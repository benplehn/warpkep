#pragma once


namespace warpkep{
    

    template <typename T>
    struct Vector3SoAConstView{
        const T* x{};
        const T* y{};
        const T* z{};
    };


    template <typename T>
    struct Vector3SoAView {
        T* x{};
        T* y{};
        T* z{};
    };


} // namespace warpkep