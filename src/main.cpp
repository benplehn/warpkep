#include <warpkep/version.hpp>

#include <iostream>

int main() {
    std::cout << "warpkep: CPU development environment ready\n";
    std::cout << "warpkep version: " << warpkep::version() << '\n';
    return 0;
}
