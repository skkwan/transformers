#include "myproject.h"

#include <iostream>
#include <fstream>

float gelu_function_float(float in) {
    return in * 0.5 * (1 + std::erf(in / std::sqrt(2)));
}

template <class data_T, typename CONFIG_T> void init_gelu_table(data_T table_out[CONFIG_T::table_size]) {

    // To-do: Try to have the GELU table be a header file generated in the tb,
    // or explore creating the GELU table on-the-fly in the gelu function.
    std::ofstream myfile;
    myfile.open("gelu_table_temp.h");
    myfile << "table_t gelu_table_temp[" << CONFIG_T::table_size << "] = {";
    for (int ii = 0; ii < CONFIG_T::table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(CONFIG_T::table_size) / 2.0) / float(CONFIG_T::table_size);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = gelu_function_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
        myfile << table_out[ii]; 
        if (ii < (CONFIG_T::table_size - 1))
            myfile << ", " << std::endl;
    }
    myfile << "};";
}


int main () {

    // Get values for the table. Actually the init_gelu_table function writes everything to a .h file, but we have the option of
    // manipulating this as a C++ object as well, e.g. as an input to the top-level function (to be decided).
    table_t gelu_table_object[TABLE_SIZE];

    init_gelu_table<table_t, gelu_config2>(gelu_table_object);

    float target_max_rmsError = 0.2;

    input_t myInput[N_INPUT_1_1] = {-3.0, -1.0, 0.0, 1.0, 3.0};
    result_t myOutput[N_INPUT_1_1];

    result_t goldenOutput[N_INPUT_1_1] = {-0.00404969409489031, -0.15865525393145707, 0.0, 0.8413447460685429, 2.99595030590511};

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
