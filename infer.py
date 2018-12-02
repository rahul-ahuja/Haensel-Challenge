import tensorflow as tf
from sklearn.model_selection import train_test_split
from analysis import processed_data, labels
from helper import one_hot_encoding, get_batches
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

X_train, X_validation, y_train, y_validation = train_test_split(processed_data, labels,
                                                                random_state=0, test_size=0.75, stratify=labels)

y_train = one_hot_encoding(y_train)

BATCH_SIZE = 128
if __name__ == '__main__':

    with tf.gfile.GFile('graph_optimized.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    G = tf.Graph()

    #tensorflow graph for inference
    with tf.Session(graph=G) as sess:

        y = tf.import_graph_def(graph_def_optimized, return_elements=['logits:0'])
        print('Operations in Optimized Graph:')
        print([op.name for op in G.get_operations()])
        feat = G.get_tensor_by_name('import/features:0')
        dropout = G.get_tensor_by_name('import/dropout:0')

        total_pred = []
        total_actual = []
        step = 0
        for batch_x, batch_y in get_batches(X_train, y_train, BATCH_SIZE):
            prediction = sess.run(y, feed_dict={feat: batch_x, dropout:1.0})
            total_actual.append(np.array(batch_y.idxmax(axis=1)))
            total_pred.append(np.array(prediction).ravel())


        classes = (np.concatenate(total_actual).ravel(), np.concatenate(total_pred).ravel())
        print(confusion_matrix(*classes))
        print()
        print(precision_recall_fscore_support(*classes, average ='weighted'))
