from tensorflow.keras.layers import Conv2D, Add, concatenate


class Rdb:
    def __init__(self, padding: str = 'same', k_size: int = 3,
                 k_init: str = 'Orthogonal', act: str = 'relu'):
        super(Rdb, self).__init__()

        self.k_size = k_size
        self.padding = padding
        self.act = act
        self.k_init = k_init
        self.add = Add()


    def get_layer(self, inputs):
        conv0 = Conv2D(filters=inputs.get_shape()[-1], kernel_size=self.k_size, padding=self.padding,
                       activation=self.act, kernel_initializer=self.k_init)(inputs)
        concat0 = concatenate([inputs, conv0])

        conv1 = Conv2D(filters=concat0.get_shape()[-1], kernel_size=self.k_size, padding=self.padding,
                       activation=self.act, kernel_initializer=self.k_init)(concat0)
        concat1 = concatenate([inputs, conv0, conv1])

        conv2 = Conv2D(filters=concat1.get_shape()[-1], kernel_size=self.k_size, padding=self.padding,
                       activation=self.act, kernel_initializer=self.k_init)(concat1)
        concat2 = concatenate([inputs, conv0, conv1, conv2])

        conv_1x1 = Conv2D(filters=inputs.get_shape()[-1], kernel_size=self.k_size,
                          padding=self.padding, activation=self.act, kernel_initializer=self.k_init)(concat2)

        return self.add([inputs, conv_1x1])


