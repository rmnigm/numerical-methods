#include <iostream>
#include <limits>

int main() {
    int k = 0;
    float a = 1;
    double b = 1;
    long double c = 1;

    while(a != 0){
        a /= 2;
        k++;
    }
    std::cout << "float zero = 2^-" << k << std::endl;
    k = 0;
    a = 1;
    while(a < std::numeric_limits<float>::max()){
        a *= 2;
        k++;
    }
    std::cout << "float infinity = 2^" << k << std::endl;
    k = 0;
    a = 1;
    while(1 + a > 1){
        a /= 2;
        k++;
    }
    std::cout << "float epsilon = 2^-" << k << std::endl;

    k = 0;
    while(b != 0){
        b /= 2;
        k++;
    }
    std::cout << "double zero = 2^-" << k << std::endl;
    k = 0;
    b = 1;
    while(b < std::numeric_limits<double>::max()){
        b *= 2;
        k++;
    }
    std::cout << "double infinity = 2^" << k << std::endl;
    k = 0;
    b = 1;
    while(1 + b > 1){
        b /= 2;
        k++;
    }
    std::cout << "double epsilon = 2^-" << k << std::endl;

    k = 0;
    while(c != 0){
        c /= 2;
        k++;
    }
    std::cout << "long double zero = 2^-" << k << std::endl;
    k = 0;
    c = 1;
    while(c < std::numeric_limits<long double>::max()){
        c *= 2;
        k++;
    }
    std::cout << "long double infinity = 2^" << k << std::endl;
    k = 0;
    c = 1;
    while(1 + c > 1){
        c /= 2;
        k++;
    }
    std::cout << "long double epsilon = 2^-" << k << std::endl;
    return 0;
}
