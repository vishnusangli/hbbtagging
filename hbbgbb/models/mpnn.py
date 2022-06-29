import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.edge_dim, self.node_dim * self.node_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.node_dim * self.node_dim),
            initializer="zeros",
            name="bias",
        )
        self.built = True

    def call(self, inputs):
        node_features, edge_features, pair_indices = inputs

        # Apply linear transformation to edge features
        edge_features = tf.matmul(edge_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        edge_features = tf.reshape(edge_features, (-1, self.node_dim, self.node_dim))

        # Obtain node features of neighbors
        node_features_neighbors = tf.gather(node_features, pair_indices[:, 1])
        node_features_neighbors = tf.expand_dims(node_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(edge_features, node_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(node_features)[0],
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.node_dim)
        self.update_step = layers.GRUCell(self.node_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        node_features, edge_features, pair_indices = inputs

        # Pad node features if number of desired units exceeds node_features dim.
        # Alternatively, a dense layer could be used here.
        node_features_updated = tf.pad(node_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            node_features_aggregated = self.message_step(
                [node_features_updated, edge_features, pair_indices]
            )

            # Update node state via a step of GRU
            node_features_updated, _ = self.update_step(
                node_features_aggregated, node_features_updated
            )
        return node_features_updated


class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):

        node_features, molecule_indicator = inputs

        # Obtain subgraphs
        node_features_partitioned = tf.dynamic_partition(
            node_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_nodes = [tf.shape(f)[0] for f in node_features_partitioned]
        max_num_nodes = tf.reduce_max(num_nodes)
        node_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_nodes - n), (0, 0)])
                for f, n in zip(node_features_partitioned, num_nodes)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(node_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(node_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)


def MPNNModel(
    node_dim,
    edge_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):

    node_features = layers.Input((node_dim), dtype="float32", name="node_features")
    edge_features = layers.Input((edge_dim), dtype="float32", name="edge_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [node_features, edge_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[node_features, edge_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model
    