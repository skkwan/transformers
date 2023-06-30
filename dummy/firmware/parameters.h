#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h" 
#include "nnet_utils/nnet_helpers.h" 
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"

// hls-fpga-machine-learning insert weights

// hls-fpga-machine-learning insert layer-config
// gelu1
struct gelu_config2 : nnet::activ_config {
    static const unsigned n_in = 5;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef gelu1_table_t table_t;
};


#endif
