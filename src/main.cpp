#include <iostream>
#include<warpkep/version.hpp>


int main() {
    std::cout << "warpkep: CPU development environment ready" << std::endl;
    std::cout << "warpkep version: " << warpkep::version() << std::endl;
    return 0;
}