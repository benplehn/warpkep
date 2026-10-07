#include <array>
#include <iostream>

struct CartesianState {
    std::array<double, 3> position;
    std::array<double, 3> velocity;
};


int main () {

    // Create a CartesianState object with position and velocity
    const CartesianState initial = {
        {1.0, 0.0, 0.0}, // position
        {0.0, 1.0, 0.0}  // velocity
    };

    // Print the initial position and velocity
    std::cout << "Initial position: ";
    for (const double component : initial.position) {
        std::cout <<' ' << component;
    }
    std::cout << "\nInitial velocity: ";
    for (const double component : initial.velocity) {
        std::cout <<' ' << component;
    }
    
    // Print a newline at the end
    std::cout << std::endl;
    return 0;


}