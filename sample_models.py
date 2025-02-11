from sys import implementation
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, GRU, LSTM, 
    MaxPooling1D, Dropout, BatchNormalization)
from keras import regularizers


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization()(rnn)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='tanh',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    rnn = GRU(units, activation='tanh',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization()(rnn)   
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def dilated_double_cnn_rnn_model(input_dim, filters, kernel_size,
    conv_border_mode, units, dilation, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d_1 = Conv1D(filters, kernel_size, 
                     padding=conv_border_mode,
                     activation='tanh',
                     dilation_rate=dilation,
                     kernel_regularizer=regularizers.l2(0.001),
                     name='conv_1d_1')(input_data)
    
    second_kernel_size = round(kernel_size * 2)
    second_filter_count = round(filters / 2)

    conv_1d_2 = Conv1D(second_filter_count, second_kernel_size, 
                     padding=conv_border_mode,
                     activation='tanh',
                     dilation_rate=dilation,
                     kernel_regularizer=regularizers.l2(0.001),
                     name='conv_1d_2')(conv_1d_1)

    rnn = GRU(
      units, 
      activation='tanh',
      return_sequences=True, 
      kernel_regularizer=regularizers.l2(0.001),
      implementation=2, name='rnn')(conv_1d_2)

    bn_rnn = BatchNormalization()(rnn)   
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
      cnn_output_length(x, kernel_size, conv_border_mode, 1, dilation=dilation),
      second_kernel_size, conv_border_mode, 1, dilation)

    print(model.summary())
    return model



def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    rnn1 = GRU(units, activation='tanh',
        return_sequences=True, implementation=2, name='rnn_1')(input_data)
    bn_rnn1 = BatchNormalization()(rnn1)
    rnn2 = GRU(units, activation='tanh',
        return_sequences=True, implementation=2, name='rnn_2')(bn_rnn1)
    bn_rnn2 = BatchNormalization()(rnn2)
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    bd_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2), name="bidi")(input_data)
    bn_bd_rnn = BatchNormalization()(bd_rnn)   
    time_dense = TimeDistributed(Dense(output_dim))(bn_bd_rnn)    
    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model():
    """ Build a deep network for speech 
    """
    units = 200
    input_dim = 161 # Spectrogram, use 13 for MFCC
    filters = 200
    kernel_size=11
    conv_stride=2
    conv_border_mode='valid'
    output_dim = 29

    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='tanh',
                     name='conv1d')(input_data)
    bn_conv = BatchNormalization()(conv_1d)
    bd_rnn = Bidirectional(GRU(units, return_sequences=True, implementation=2), name="bidi")(bn_conv)
    bn_bd_rnn = BatchNormalization()(bd_rnn)  
    time_dense = TimeDistributed(Dense(output_dim))(bn_bd_rnn)    
    dropout = Dropout(0.1)(time_dense)
    y_pred = Activation('softmax', name='softmax')(dropout)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
