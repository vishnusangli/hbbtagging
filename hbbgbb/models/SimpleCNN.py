import sonnet as snt
import tensorflow as tf

class CNNModel(snt.Module):
    def __init__(self, name = "cnn"):
        super(CNNModel, self).__init__(name=name)
        self.norm = snt.BatchNorm(True, True)
        #Need to look into the desired output shape - padding and output channels
        self.hidden1 = snt.Conv2D(output_channels=15, kernel_shape=(3, 3), padding="SAME", name = 'hconv_1')
        self.hidden2 = snt.Conv2D(output_channels=10, kernel_shape=(3,3), padding="SAME", name = "hconv_2")
        self.flatten = snt.Flatten()
        self.linear = snt.Linear(3)
    def __call__(self, data, is_training=False):
        output = self.norm(data, is_training=is_training)

        output = tf.nn.relu(self.hidden1(data))
        output = tf.nn.relu(self.hidden2(data))

        output = self.flatten(output)
        output = self.linear(output)
        return tf.nn.softmax(output)
