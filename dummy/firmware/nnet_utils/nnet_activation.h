#ifndef NNET_ACTIVATION_H_
#define NNET_ACTIVATION_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include "hls_math.h"
#include <cmath>

namespace nnet {

struct activ_config {
    // IO size
    static const unsigned n_in = 10;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<18, 8> table_t;
};

// *************************************************
//       LINEAR Activation -- See Issue 53
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void linear(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        res[ii] = data[ii];
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void relu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

template <class data_T, class res_T, int MAX_INT, typename CONFIG_T>
void relu_max(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg < 0)
            res[ii] = 0;
        else if (datareg > MAX_INT)
            res[ii] = MAX_INT;
        else
            res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void relu6(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 6, CONFIG_T>(data, res);
}

template <class data_T, class res_T, typename CONFIG_T> void relu1(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    relu_max<data_T, res_T, 1, CONFIG_T>(data, res);
}

// *************************************************
//       GELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T> void gelu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {

    ap_fixed<18,8> gelu_table[1024] = {-0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.000976563, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00195313, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00292969, 
    -0.00390625, 
    -0.00390625, 
    -0.00390625, 
    -0.00390625, 
    -0.00390625, 
    -0.00390625, 
    -0.00488281, 
    -0.00488281, 
    -0.00488281, 
    -0.00488281, 
    -0.00488281, 
    -0.00585938, 
    -0.00585938, 
    -0.00585938, 
    -0.00585938, 
    -0.00683594, 
    -0.00683594, 
    -0.00683594, 
    -0.0078125, 
    -0.0078125, 
    -0.0078125, 
    -0.00878906, 
    -0.00878906, 
    -0.00878906, 
    -0.00976563, 
    -0.00976563, 
    -0.00976563, 
    -0.0107422, 
    -0.0107422, 
    -0.0117188, 
    -0.0117188, 
    -0.0126953, 
    -0.0126953, 
    -0.0136719, 
    -0.0136719, 
    -0.0146484, 
    -0.0146484, 
    -0.015625, 
    -0.015625, 
    -0.0166016, 
    -0.0175781, 
    -0.0175781, 
    -0.0185547, 
    -0.0195313, 
    -0.0195313, 
    -0.0205078, 
    -0.0214844, 
    -0.0224609, 
    -0.0224609, 
    -0.0234375, 
    -0.0244141, 
    -0.0253906, 
    -0.0263672, 
    -0.0273438, 
    -0.0283203, 
    -0.0292969, 
    -0.0302734, 
    -0.03125, 
    -0.0322266, 
    -0.0332031, 
    -0.0341797, 
    -0.0351563, 
    -0.0361328, 
    -0.0371094, 
    -0.0380859, 
    -0.0400391, 
    -0.0410156, 
    -0.0419922, 
    -0.0429688, 
    -0.0449219, 
    -0.0458984, 
    -0.046875, 
    -0.0488281, 
    -0.0498047, 
    -0.0517578, 
    -0.0527344, 
    -0.0546875, 
    -0.0556641, 
    -0.0576172, 
    -0.0585938, 
    -0.0605469, 
    -0.0625, 
    -0.0634766, 
    -0.0654297, 
    -0.0673828, 
    -0.0693359, 
    -0.0703125, 
    -0.0722656, 
    -0.0742188, 
    -0.0761719, 
    -0.078125, 
    -0.0791016, 
    -0.0810547, 
    -0.0830078, 
    -0.0849609, 
    -0.0869141, 
    -0.0888672, 
    -0.0908203, 
    -0.0927734, 
    -0.0947266, 
    -0.0966797, 
    -0.0986328, 
    -0.100586, 
    -0.102539, 
    -0.104492, 
    -0.106445, 
    -0.108398, 
    -0.110352, 
    -0.112305, 
    -0.115234, 
    -0.117188, 
    -0.119141, 
    -0.121094, 
    -0.123047, 
    -0.125, 
    -0.126953, 
    -0.128906, 
    -0.130859, 
    -0.132813, 
    -0.134766, 
    -0.136719, 
    -0.138672, 
    -0.139648, 
    -0.141602, 
    -0.143555, 
    -0.145508, 
    -0.147461, 
    -0.148438, 
    -0.150391, 
    -0.152344, 
    -0.15332, 
    -0.155273, 
    -0.15625, 
    -0.158203, 
    -0.15918, 
    -0.160156, 
    -0.162109, 
    -0.163086, 
    -0.164063, 
    -0.165039, 
    -0.166016, 
    -0.166992, 
    -0.166992, 
    -0.167969, 
    -0.168945, 
    -0.168945, 
    -0.169922, 
    -0.169922, 
    -0.169922, 
    -0.170898, 
    -0.170898, 
    -0.169922, 
    -0.169922, 
    -0.169922, 
    -0.169922, 
    -0.168945, 
    -0.167969, 
    -0.167969, 
    -0.166992, 
    -0.166016, 
    -0.165039, 
    -0.163086, 
    -0.162109, 
    -0.160156, 
    -0.158203, 
    -0.157227, 
    -0.154297, 
    -0.152344, 
    -0.150391, 
    -0.147461, 
    -0.145508, 
    -0.142578, 
    -0.139648, 
    -0.136719, 
    -0.132813, 
    -0.129883, 
    -0.125977, 
    -0.12207, 
    -0.118164, 
    -0.114258, 
    -0.110352, 
    -0.105469, 
    -0.100586, 
    -0.0957031, 
    -0.0908203, 
    -0.0859375, 
    -0.0800781, 
    -0.0742188, 
    -0.0693359, 
    -0.0625, 
    -0.0566406, 
    -0.0507813, 
    -0.0439453, 
    -0.0371094, 
    -0.0302734, 
    -0.0234375, 
    -0.015625, 
    -0.0078125, 
    0, 
    0.0078125, 
    0.015625, 
    0.0234375, 
    0.0322266, 
    0.0410156, 
    0.0498047, 
    0.0585938, 
    0.0683594, 
    0.078125, 
    0.0869141, 
    0.0976563, 
    0.107422, 
    0.117188, 
    0.12793, 
    0.138672, 
    0.149414, 
    0.160156, 
    0.170898, 
    0.182617, 
    0.194336, 
    0.206055, 
    0.217773, 
    0.229492, 
    0.242188, 
    0.253906, 
    0.266602, 
    0.279297, 
    0.291992, 
    0.305664, 
    0.318359, 
    0.332031, 
    0.345703, 
    0.358398, 
    0.373047, 
    0.386719, 
    0.400391, 
    0.415039, 
    0.428711, 
    0.443359, 
    0.458008, 
    0.472656, 
    0.488281, 
    0.50293, 
    0.517578, 
    0.533203, 
    0.548828, 
    0.564453, 
    0.579102, 
    0.594727, 
    0.611328, 
    0.626953, 
    0.642578, 
    0.65918, 
    0.674805, 
    0.691406, 
    0.708008, 
    0.723633, 
    0.740234, 
    0.756836, 
    0.773438, 
    0.790039, 
    0.806641, 
    0.824219, 
    0.84082, 
    0.857422, 
    0.875, 
    0.891602, 
    0.90918, 
    0.925781, 
    0.943359, 
    0.960938, 
    0.977539, 
    0.995117, 
    1.0127, 
    1.03027, 
    1.04785, 
    1.06445, 
    1.08203, 
    1.09961, 
    1.11719, 
    1.13477, 
    1.15234, 
    1.16992, 
    1.1875, 
    1.20508, 
    1.22266, 
    1.24023, 
    1.25781, 
    1.27539, 
    1.29395, 
    1.31152, 
    1.3291, 
    1.34668, 
    1.36426, 
    1.38184, 
    1.39941, 
    1.41699, 
    1.43457, 
    1.45215, 
    1.46973, 
    1.4873, 
    1.50488, 
    1.52246, 
    1.54004, 
    1.55762, 
    1.5752, 
    1.59277, 
    1.60938, 
    1.62695, 
    1.64453, 
    1.66211, 
    1.67969, 
    1.69629, 
    1.71387, 
    1.73145, 
    1.74902, 
    1.76563, 
    1.7832, 
    1.80078, 
    1.81738, 
    1.83496, 
    1.85156, 
    1.86914, 
    1.88574, 
    1.90332, 
    1.91992, 
    1.9375, 
    1.9541, 
    1.9707, 
    1.98828, 
    2.00488, 
    2.02148, 
    2.03809, 
    2.05566, 
    2.07227, 
    2.08887, 
    2.10547, 
    2.12207, 
    2.13867, 
    2.15527, 
    2.17188, 
    2.18848, 
    2.20508, 
    2.22168, 
    2.23828, 
    2.25488, 
    2.27148, 
    2.28809, 
    2.30469, 
    2.32129, 
    2.33691, 
    2.35352, 
    2.37012, 
    2.38672, 
    2.40234, 
    2.41895, 
    2.43555, 
    2.45117, 
    2.46777, 
    2.48438, 
    2.5, 
    2.5166, 
    2.53223, 
    2.54883, 
    2.56445, 
    2.58105, 
    2.59668, 
    2.61328, 
    2.62891, 
    2.64551, 
    2.66113, 
    2.67773, 
    2.69336, 
    2.70898, 
    2.72559, 
    2.74121, 
    2.75684, 
    2.77344, 
    2.78906, 
    2.80469, 
    2.82129, 
    2.83691, 
    2.85254, 
    2.86914, 
    2.88477, 
    2.90039, 
    2.91602, 
    2.93262, 
    2.94824, 
    2.96387, 
    2.97949, 
    2.99512, 
    3.01172, 
    3.02734, 
    3.04297, 
    3.05859, 
    3.07422, 
    3.08984, 
    3.10645, 
    3.12207, 
    3.1377, 
    3.15332, 
    3.16895, 
    3.18457, 
    3.2002, 
    3.21582, 
    3.23145, 
    3.24805, 
    3.26367, 
    3.2793, 
    3.29492, 
    3.31055, 
    3.32617, 
    3.3418, 
    3.35742, 
    3.37305, 
    3.38867, 
    3.4043, 
    3.41992, 
    3.43555, 
    3.45215, 
    3.46777, 
    3.4834, 
    3.49902, 
    3.51465, 
    3.53027, 
    3.5459, 
    3.56152, 
    3.57715, 
    3.59277, 
    3.6084, 
    3.62402, 
    3.63965, 
    3.65527, 
    3.6709, 
    3.68652, 
    3.70215, 
    3.71777, 
    3.7334, 
    3.74902, 
    3.76465, 
    3.78027, 
    3.7959, 
    3.81152, 
    3.82715, 
    3.84277, 
    3.8584, 
    3.87402, 
    3.88965, 
    3.90527, 
    3.9209, 
    3.93652, 
    3.95215, 
    3.96777, 
    3.9834, 
    3.99902, 
    4.01465, 
    4.03027, 
    4.0459, 
    4.06152, 
    4.07715, 
    4.09277, 
    4.1084, 
    4.12402, 
    4.13965, 
    4.15527, 
    4.1709, 
    4.18652, 
    4.20215, 
    4.21777, 
    4.2334, 
    4.24902, 
    4.26465, 
    4.28027, 
    4.2959, 
    4.31152, 
    4.32715, 
    4.34277, 
    4.3584, 
    4.37402, 
    4.38965, 
    4.40527, 
    4.4209, 
    4.43652, 
    4.45215, 
    4.46777, 
    4.4834, 
    4.49902, 
    4.51465, 
    4.53027, 
    4.5459, 
    4.56152, 
    4.57715, 
    4.59277, 
    4.6084, 
    4.62402, 
    4.63965, 
    4.65527, 
    4.6709, 
    4.68652, 
    4.70215, 
    4.71777, 
    4.7334, 
    4.74902, 
    4.76465, 
    4.78027, 
    4.7959, 
    4.81152, 
    4.82715, 
    4.84277, 
    4.8584, 
    4.87402, 
    4.88965, 
    4.90527, 
    4.9209, 
    4.93652, 
    4.95215, 
    4.96777, 
    4.9834, 
    4.99902, 
    5.01465, 
    5.03027, 
    5.0459, 
    5.06152, 
    5.07715, 
    5.09277, 
    5.1084, 
    5.12402, 
    5.13965, 
    5.15527, 
    5.1709, 
    5.18652, 
    5.20215, 
    5.21777, 
    5.2334, 
    5.24902, 
    5.26465, 
    5.28027, 
    5.2959, 
    5.31152, 
    5.32715, 
    5.34277, 
    5.35938, 
    5.375, 
    5.39063, 
    5.40625, 
    5.42188, 
    5.4375, 
    5.45313, 
    5.46875, 
    5.48438, 
    5.5, 
    5.51563, 
    5.53125, 
    5.54688, 
    5.5625, 
    5.57813, 
    5.59375, 
    5.60938, 
    5.625, 
    5.64063, 
    5.65625, 
    5.67188, 
    5.6875, 
    5.70313, 
    5.71875, 
    5.73438, 
    5.75, 
    5.76563, 
    5.78125, 
    5.79688, 
    5.8125, 
    5.82813, 
    5.84375, 
    5.85938, 
    5.875, 
    5.89063, 
    5.90625, 
    5.92188, 
    5.9375, 
    5.95313, 
    5.96875, 
    5.98438, 
    6, 
    6.01563, 
    6.03125, 
    6.04688, 
    6.0625, 
    6.07813, 
    6.09375, 
    6.10938, 
    6.125, 
    6.14063, 
    6.15625, 
    6.17188, 
    6.1875, 
    6.20313, 
    6.21875, 
    6.23438, 
    6.25, 
    6.26563, 
    6.28125, 
    6.29688, 
    6.3125, 
    6.32813, 
    6.34375, 
    6.35938, 
    6.375, 
    6.39063, 
    6.40625, 
    6.42188, 
    6.4375, 
    6.45313, 
    6.46875, 
    6.48438, 
    6.5, 
    6.51563, 
    6.53125, 
    6.54688, 
    6.5625, 
    6.57813, 
    6.59375, 
    6.60938, 
    6.625, 
    6.64063, 
    6.65625, 
    6.67188, 
    6.6875, 
    6.70313, 
    6.71875, 
    6.73438, 
    6.75, 
    6.76563, 
    6.78125, 
    6.79688, 
    6.8125, 
    6.82813, 
    6.84375, 
    6.85938, 
    6.875, 
    6.89063, 
    6.90625, 
    6.92188, 
    6.9375, 
    6.95313, 
    6.96875, 
    6.98438, 
    7, 
    7.01563, 
    7.03125, 
    7.04688, 
    7.0625, 
    7.07813, 
    7.09375, 
    7.10938, 
    7.125, 
    7.14063, 
    7.15625, 
    7.17188, 
    7.1875, 
    7.20313, 
    7.21875, 
    7.23438, 
    7.25, 
    7.26563, 
    7.28125, 
    7.29688, 
    7.3125, 
    7.32813, 
    7.34375, 
    7.35938, 
    7.375, 
    7.39063, 
    7.40625, 
    7.42188, 
    7.4375, 
    7.45313, 
    7.46875, 
    7.48438, 
    7.5, 
    7.51563, 
    7.53125, 
    7.54688, 
    7.5625, 
    7.57813, 
    7.59375, 
    7.60938, 
    7.625, 
    7.64063, 
    7.65625, 
    7.67188, 
    7.6875, 
    7.70313, 
    7.71875, 
    7.73438, 
    7.75, 
    7.76563, 
    7.78125, 
    7.79688, 
    7.8125, 
    7.82813, 
    7.84375, 
    7.85938, 
    7.875, 
    7.89063, 
    7.90625, 
    7.92188, 
    7.9375, 
    7.95313, 
    7.96875, 
    7.98438};


    #pragma HLS PIPELINE

    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0) 
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T) gelu_table[index];
    }
}   

// *************************************************
//       Sigmoid Activation
// *************************************************
inline float sigmoid_fcn_float(float input) { return 1.0 / (1 + std::exp(-input)); }

template <typename CONFIG_T, int N_TABLE> void init_sigmoid_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default logistic sigmoid function:
    //   result = 1/(1+e^(-x))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = sigmoid_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)sigmoid_table[index];
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

enum class softmax_implementation { latency = 0, legacy = 1, stable = 2, argmax = 3 };

inline float exp_fcn_float(float input) { return std::exp(input); }

template <class data_T, typename CONFIG_T> inline float softmax_real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    data_T x(0);
    x(x.width - 1, x.width - N) = i;
    return (float)x;
}

template <class data_T, typename CONFIG_T> inline unsigned softmax_idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    ap_uint<N> y = x(x.width - 1, x.width - N);              // slice the top N bits of input
    return (unsigned)y(N - 1, 0);
}

template <class data_T, typename CONFIG_T>
void init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
        // Slicing bits for address is going to round towards 0, so take the central value
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::exp_table_t exp_x = exp_fcn_float(x);
        table_out[i] = exp_x;
    }
}

template <class data_T, typename CONFIG_T>
void init_invert_table(typename CONFIG_T::inv_table_t table_out[CONFIG_T::table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
        float x = softmax_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::inv_table_t inv_x = 1 / x;
        table_out[i] = inv_x;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS pipeline
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS unroll
        unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(data[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS pipeline
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    // Find the max and compute all delta(x_i, x_max)
    Op_max<data_T> op_max;
    data_T x_max = reduce<data_T, CONFIG_T::n_in, Op_max<data_T>>(data, op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    ap_fixed<data_T::width, data_T::iwidth, AP_RND, AP_SAT> d_xi_xmax[CONFIG_T::n_in];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS unroll
        d_xi_xmax[i] = data[i] - x_max;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS unroll
        unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(d_xi_xmax[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum =
        reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum =
        invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t, CONFIG_T>(exp_sum)];
    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
    }
}

template <typename CONFIG_T, int N_TABLE> void init_exp_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = exp_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <typename CONFIG_T, int N_TABLE> void init_invert_table_legacy(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Inversion function:
    //   result = 1/x
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +64)
        float in_val = 64.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0)
            table_out[ii] = 1.0 / in_val;
        else
            table_out[ii] = 0.0;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in]; // different, independent, fixed point precision
    typename CONFIG_T::table_t exp_diff_res;            // different, independent, fixed point precision
    data_T data_cache[CONFIG_T::n_in];
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_cache[ii] = data[ii];
        exp_res[ii] = 0;
    }

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        for (int jj = 0; jj < CONFIG_T::n_in; jj++) {
            if (ii == jj)
                exp_diff_res = 1;
            else {
                data_round = (data_cache[jj] - data_cache[ii]) * CONFIG_T::table_size / 16;
                index = data_round + 8 * CONFIG_T::table_size / 16;
                if (index < 0)
                    index = 0;
                if (index > CONFIG_T::table_size - 1)
                    index = CONFIG_T::table_size - 1;
                exp_diff_res = exp_table[index];
            }
            exp_res[ii] += exp_diff_res;
        }
    }

    // Second loop to invert
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        int exp_res_index = exp_res[ii] * CONFIG_T::table_size / 64;
        if (exp_res_index < 0)
            exp_res_index = 0;
        if (exp_res_index > CONFIG_T::table_size - 1)
            exp_res_index = CONFIG_T::table_size - 1;
        // typename CONFIG_T::table_t exp_res_invert = invert_table[exp_res_index];
        res[ii] = (res_T)invert_table[exp_res_index];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        res[i] = (res_T)0;
    }

    data_T maximum = data[0];
    int idx = 0;

    for (int i = 1; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE
        if (data[i] > maximum) {
            maximum = data[i];
            idx = i;
        }
    }

    res[idx] = (res_T)1;
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS inline
    switch (CONFIG_T::implementation) {
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::argmax:
        softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
        break;
    }
}

// *************************************************
//       TanH Activation
// *************************************************
template <typename CONFIG_T, int N_TABLE> void init_tanh_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Implement tanh lookup
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -4 to +4)
        float in_val = 2 * 4.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = tanh(in_val);
        // std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val <<
        // std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t tanh_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_tanh_table<CONFIG_T, CONFIG_T::table_size>(tanh_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 8;
        index = data_round + 4 * CONFIG_T::table_size / 8;
        // std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)tanh_table[index];
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto datareg = CONFIG_T::slope * data[ii] + CONFIG_T::shift;
        if (datareg > 1)
            datareg = 1;
        else if (datareg < 0)
            datareg = 0;
        res[ii] = datareg;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void hard_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    if (CONFIG_T::io_type == io_parallel) {
        #pragma HLS PIPELINE
    }

    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        auto sigmoid = CONFIG_T::slope * data[ii] + CONFIG_T::shift;
        if (sigmoid > 1)
            sigmoid = 1;
        else if (sigmoid < 0)
            sigmoid = 0;
        res[ii] = 2 * sigmoid - 1;
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void leaky_relu(data_T data[CONFIG_T::n_in], data_T alpha, res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha * datareg;
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void thresholded_relu(data_T data[CONFIG_T::n_in], data_T theta, res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > theta)
            res[ii] = datareg;
        else
            res[ii] = 0;
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
inline float softplus_fcn_float(float input) { return std::log(std::exp(input) + 1.); }

template <typename CONFIG_T, int N_TABLE> void init_softplus_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default softplus function:
    //   result = log(exp(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softplus_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softplus(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softplus_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softplus_table<CONFIG_T, CONFIG_T::table_size>(softplus_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)softplus_table[index];
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
inline float softsign_fcn_float(float input) { return input / (std::abs(input) + 1.); }

template <typename CONFIG_T, int N_TABLE> void init_softsign_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default softsign function:
    //   result = x / (abs(x) + 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = softsign_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softsign(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t softsign_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_softsign_table<CONFIG_T, CONFIG_T::table_size>(softsign_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    // Index into the lookup table based on data
    int data_round;
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_round = data[ii] * CONFIG_T::table_size / 16;
        index = data_round + 8 * CONFIG_T::table_size / 16;
        if (index < 0)
            index = 0;
        if (index > CONFIG_T::table_size - 1)
            index = CONFIG_T::table_size - 1;
        res[ii] = (res_T)softsign_table[index];
    }
}

// *************************************************
//       ELU Activation
// *************************************************
inline float elu_fcn_float(float input) { return std::exp(input) - 1.; }

template <typename CONFIG_T, int N_TABLE> void init_elu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default ELU function:
    //   result = alpha * (e^(x) - 1)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = elu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void elu(data_T data[CONFIG_T::n_in], const res_T alpha, res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t elu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_elu_table<CONFIG_T, CONFIG_T::table_size>(elu_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = datareg;
        } else {
            index = datareg * CONFIG_T::table_size / -8;
            if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            res[ii] = alpha * elu_table[index];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T> void elu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SELU Activation
// *************************************************
inline float selu_fcn_float(float input) {
    return 1.0507009873554804934193349852946 * (1.6732632423543772848170429916717 * (std::exp(input) - 1.));
}

template <typename CONFIG_T, int N_TABLE> void init_selu_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    // Default SELU function:
    //   result = 1.05 * (1.673 * (e^(x) - 1))
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to 0)
        float in_val = -8.0 * ii / float(N_TABLE);
        // Next, compute lookup table function
        typename CONFIG_T::table_t real_val = selu_fcn_float(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template <class data_T, class res_T, typename CONFIG_T> void selu(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t selu_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_selu_table<CONFIG_T, CONFIG_T::table_size>(selu_table);
        initialized = true;
    }

    #pragma HLS PIPELINE

    data_T datareg;
    // Index into the lookup table based on data
    int index;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg >= 0) {
            res[ii] = res_T(1.0507009873554804934193349852946) * datareg;
        } else {
            index = datareg * CONFIG_T::table_size / -8;
            if (index > CONFIG_T::table_size - 1)
                index = CONFIG_T::table_size - 1;
            res[ii] = selu_table[index];
        }
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void prelu(data_T data[CONFIG_T::n_in], data_T alpha[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            res[ii] = datareg;
        else
            res[ii] = alpha[ii] * datareg;
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void binary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    res_T cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        if (datareg > 0)
            cache = 1;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    #pragma HLS PIPELINE

    data_T datareg;
    res_T cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = 2 * data[ii];
        if (datareg > 1)
            cache = 1;
        else if (datareg > -1 && datareg <= 1)
            cache = 0;
        else
            cache = -1;

        res[ii] = (res_T)cache;
    }
}

} // namespace nnet

#endif
