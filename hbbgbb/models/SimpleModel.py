import sonnet as snt
import tensorflow as tf

class SimpleModel(snt.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.norm = snt.BatchNorm(True, True)
        self.hidden1 = snt.Linear(1024, name="hidden1")
        self.hidden2 = snt.Linear(1024, name="hidden2")
        self.logits = snt.Linear(3, name="logits")
    def __call__(self, data, is_training=False):
        output = self.norm(data, is_training=is_training)
        output = tf.nn.relu(self.hidden1(output))
        output = tf.nn.relu(self.hidden2(output))
        output = self.logits(output)
        return output