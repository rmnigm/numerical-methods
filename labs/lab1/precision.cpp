#include <iostream>

int main() {
    int k = 0;
    float a = 1;
    double b = 1;
    long double c = 1;
    while(a != 0){
        a /= 2;
        k++;
    }
    std::cout << "Float - 2^" << k << std::endl;
    k = 0;
    while(b != 0){
        b /= 2;
        k++;
    }
    std::cout << "Double - 2^" << k << std::endl;
    k = 0;
    while(c != 0){
        c /= 2;
        k++;
    }
    std::cout << "Float - 2^" << k << std::endl;
    return 0;
}

