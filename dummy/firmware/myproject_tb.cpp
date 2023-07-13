#include "myproject.h"

#include <iostream>
#include <fstream>

/*
 * Exact implementation of GELU.
 */
float gelu_function_float(float in) {
    return in * 0.5 * (1 + std::erf(in / std::sqrt(2)));
}

/*
 * Write a GELU table to a file gelu_table_temp.h. The table should be pasted into the GELU function.
 */
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

/*
 * 
 */
template<class data_T, unsigned int table_size>
void fillArrayFromTable(std::string filename, data_T (&table)[table_size]) {
    std::ifstream inFile;
    std::cout << "Before opening input " << filename << std::endl;
    inFile.open(filename, std::ifstream::in);
    std::string line;

    int nEntry = 0;

    for (; std::getline(inFile, line); ) {
        if (nEntry >= table_size) {
            break;
        }

        std::cout << ".... read line " << line << std::endl;
	    data_T data = (data_T) std::stof(line); // convert string to float, then cast to data_T
        table[nEntry] = data;
        nEntry++;
    }
}

/*
 * Test bench function.
 */
int main () {

    // // ONCE ONLY: Get values for the table. Actually the init_gelu_table function writes everything to a .h file, but we have the option of
    // // manipulating this as a C++ object as well, e.g. as an input to the top-level function (to be decided).
    // table_t gelu_table_object[TABLE_SIZE];

    // // init_gelu_table<table_t, gelu_config2>(gelu_table_object);

    float target_max_rmsError = 0.2;

    // Get array from input file
    input_t myInput[N_INPUT_1_1];
    fillArrayFromTable("input.txt", myInput);

    result_t myOutput[N_INPUT_1_1];

    // Ideal output
    result_t goldenOutput[N_INPUT_1_1];
    fillArrayFromTable("idealOut.txt", goldenOutput);

    // Execute the function 
    myproject(myInput, myOutput);

    // Compare with the golden output: compute the root-mean-square error
    float numerator = 0;
    float rmsError = 99;

    for (int i = 0; i < N_INPUT_1_1; i++) {
        // Print for our edification
        float outAsFloat = myOutput[i].to_float();       
        float goldenOutAsFloat = goldenOutput[i].to_float();  
        std::cout << "For input " << myInput[i] << ", as float: output : " << outAsFloat << ", comparing with " << goldenOutAsFloat << std::endl;

        // Increment the numerator for the RMS error
        numerator += std::pow((goldenOutAsFloat - outAsFloat), 2);

        // // Also check per-value that the percentage difference is less than 5%
        // if (std::abs((outAsFloat - goldenOutAsFloat)/goldenOutAsFloat) > 0.05) {
        //     std::cout << "Difference is greater than 5%, test failed" << std::endl;
        //     return 1;
        // }
    }
    
    rmsError = std::sqrt(numerator/N_INPUT_1_1);

    std::cout << "numerator " << numerator << ", rmsError " << rmsError << std::endl;

    int passTest;
    if (rmsError < target_max_rmsError) passTest = 0;
    else                                passTest = 1;

    return passTest;

}
