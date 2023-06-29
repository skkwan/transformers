#include "myproject.h"

int main () {

    float target_max_rmsError = 0.2;

    input_t myInput[N_INPUT_1_1] = {-3.0, -1.0, 0.0, 1.0, 3.0};
    result_t myOutput[N_INPUT_1_1];

    // [-0.00404969409489031, -0.15865525393145702, 0.0, 0.841344746068543, 2.99595030590511]
    result_t goldenOutput[N_INPUT_1_1] = {-0.002929, -0.158203, 0, 0.84028, 2.99512};

    // Execute the function 
    myproject(myInput, myOutput);

    // Compare with the golden output: compute the root-mean-square error
    int passTest;
    float numerator = 0;
    float rmsError = 99;

    for (int i = 0; i < N_INPUT_1_1; i++) {
            // Print for our edification
            float outAsFloat = myOutput[i].to_float();       
            float goldenOutAsFloat = goldenOutput[i].to_float();  
            std::cout << "For input " << myInput[i] << ", as float: output : " << outAsFloat << ", comparing with " << goldenOutAsFloat << std::endl;
            // Increment the numerator
            numerator += std::pow((goldenOutAsFloat - outAsFloat), 2);
    }
    
    rmsError = std::sqrt(numerator/N_INPUT_1_1);

    std::cout << "numerator " << numerator << ", rmsError " << rmsError << std::endl;

    if (rmsError < target_max_rmsError) passTest = 0;
    else                                passTest = 1;

    return passTest;
}
