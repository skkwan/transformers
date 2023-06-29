import numpy as np
import hls4ml
import tensorflow as tf
from tensorflow.keras.layers import Activation

def main():
    # Keras 
    input_ = tf.keras.Input(shape=(5,))
    output = Activation(activation='gelu', name='gelu1')(input_)
    model = tf.keras.Model(input_, output, name='gelu_test')
    model.summary()

    # hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir='dummy', part='xcu250-figd2104-2L-e'
    )
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
    hls_model.compile()

    # Compare the two
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

    # Keras output
    y_keras = model.predict(x).flatten()
    print(y_keras)

    # HLS output: do not predict
    # y_hls = hls_model.predict(x)
    # print(y_hls)

    hls_model.build


if __name__ == "__main__":
    main()