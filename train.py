from sklearn.model_selection import train_test_split
from analysis import processed_data, labels, X_resampled, y_resampled
from helper import one_hot_encoding, get_batches
import tensorflow as tf


#stratefied splitting the data set into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(processed_data, labels,
                                                                random_state=0, test_size=0.9, stratify=labels)

#applying one hot encoding to the class labels
y_train = one_hot_encoding(y_train)
y_validation = one_hot_encoding(y_validation)

#getting the total number of columns
num_classes = y_train.shape[1]
n_input_nodes = X_train.shape[1]

#stratefied splitting the data set generated by SMOTE into training and validation sets
#X_train, X_validation, y_train, y_validation = train_test_split(X_resampled, y_resampled,
#                                                                random_state=0, test_size=0.9, stratify=labels)





def model_inputs(input_dim, out_dim):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        input_dim: number of input nodes
        out_dim: number of class labels

    '''
    features = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="features")
    targets = tf.placeholder(dtype=tf.int32, shape=(None))
    one_hot_y = tf.one_hot(targets, out_dim)
    dropout_prob = tf.placeholder(tf.float32, name='dropout')
    return features, one_hot_y, dropout_prob


def dense_layers(x, hidden_dim, dropout_prob, n_layers=[100, 200]):
    ''' Hidden layers being stacked together
        Arguments
        ---------
        x: input features
        hidden_dim: Number of hidden nodes
        dropout_prob: rate to drop the nodes during training
        n_layers: number nodes in each list of layer.
                  by defualt 100 nodes in first layer and 200 nodes in 2nd layer
    '''
    layer = tf.layers.dense(inputs=x, units=hidden_dim, activation=tf.nn.relu, use_bias=True,
                            bias_initializer=tf.constant_initializer(0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    layer = tf.nn.dropout(layer, keep_prob=dropout_prob)

    for idx, h_nodes in enumerate(n_layers):
        layer = tf.nn.dropout(tf.layers.dense(inputs=layer, units=h_nodes, activation=tf.nn.relu, use_bias=True,
                                              bias_initializer=tf.constant_initializer(0.01),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)),
                              keep_prob=dropout_prob)
    return layer


def mlp(x, hidden_dim, out_dim, dropout_prob):
    ''' Stacking all the layers of neural net together
        Arguments
        ---------
        x: input feature
        hidden_dim: Number of hidden nodes
        out_dim: number of output nodes
        dropout_prob: rate of dropping hidden nodes during training
    '''
    last_layer = dense_layers(x, hidden_dim, dropout_prob, n_layers=[100, 200])
    logits = tf.layers.dense(inputs=last_layer, units=out_dim, activation=None)

    return logits


def model_loss(x, logits, one_hot_y):
    ''' Working on cross entropy loss function
        Arguments
        ---------
        x: input feature
        logits: linear layer before applying softmax to get output layer
        one_hot_y: one hot encoded target
    '''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    return loss_operation


def model_opt(mlp_loss):
    ''' Optimization of the parameters of neural nets
            Arguments
            ---------
            mlp_loss: MLP's loss value
    '''

    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    training_operation = optimizer.minimize(mlp_loss)
    return training_operation


class model:
    def __init__(self, input_size, output_size, hidden_size):

        self.input, self.output_one_hot, self.dropout_prob = model_inputs(input_size, output_size)
        self.mlp_logits = mlp(self.input, hidden_size, output_size, self.dropout_prob)
        self.loss_opt = model_loss(self.input, self.mlp_logits, self.output_one_hot)
        self.train_opt = model_opt(self.loss_opt)


#if __name__ == '__main__':

def train_net(net, X, y):
    ''' Tensorflow session
            Arguments
            ---------
            net: neural network class
            X: features data
            y: labels
        '''

    EPOCHS = 2
    BATCH_SIZE = 128
    #net = model(n_input_nodes, num_classes, 200)
    saver = tf.train.Saver()
    # Create a TensorFlow configuration object. This will be
    # passed as an argument to the session.
    config = tf.ConfigProto()
    # JIT level, this can be set to ON_1 or ON_2
    jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        print()
        for i in range(EPOCHS):
            for batch_x, batch_y in get_batches(X, y, BATCH_SIZE):

                sess.run(net.train_opt, feed_dict={net.input: batch_x, net.output_one_hot: batch_y, net.dropout_prob:0.75})
                loss = sess.run(net.loss_opt, feed_dict={net.input: batch_x, net.output_one_hot: batch_y, net.dropout_prob:0.75})
                #print(loss)
                prediction, actual = sess.run([tf.argmax(net.mlp_logits, 1, name='logits'), tf.argmax(batch_y, 1)],
                                              feed_dict={net.input: batch_x, net.output_one_hot: batch_y, net.dropout_prob:0.75})

            #print(prediction)

        # Save GraphDef
        tf.train.write_graph(sess.graph_def, '.', 'graph.pb')

        # Save checkpoint
        saver.save(sess=sess, save_path="test_model")

if __name__ == '__main__':
    net = model(n_input_nodes, num_classes, 200)
    #train(net, X_resampled, y_resampled)
    train_net(net, X_train, y_train)