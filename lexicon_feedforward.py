from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import data_util as du

def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='layer_weights_')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.random_normal([1,out_size]), name='layer_biases_')
        with tf.name_scope('Wx_plus_b'):
            #print (inputs.shape)
            #print (biases.shape)
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            #print (Wx_plus_b)
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#assum there is not window for input
def add_projection_layer(inputs,in_size,out_size,activation_function = None):
    print ("in projection layer",inputs.shape)
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.random_normal([1, out_size]))
    outputs = tf.nn.embedding_lookup(Weights, inputs) + biases
    print ("in projection layer", outputs.shape)
    return outputs

def set_output_gama(trg_word_list, out_size):
    result = np.zeros(len(trg_word_list) * out_size).reshape(len(trg_word_list), out_size)
    for no in range(len(trg_word_list)):
        result[no][int(trg_word_list[no])] = 1.  # here can be changed as the real gama value later!
    #print result.shape
    return result



#define network structure
vocabulary_size = 4000
projection_num = 500
hidden_num = 100
output_num = 2000

#define the inputs and outputs to the neural netwok
with tf.name_scope('Inputs'):
    xs = tf.placeholder(tf.int32,[None],name='input_data_')
    ys = tf.placeholder(tf.float32,[None,output_num],name='output_data_')

project_output_ = add_projection_layer(xs, vocabulary_size, projection_num,activation_function=None)
hidden_layer_0_ = add_layer(project_output_, projection_num,hidden_num,activation_function=tf.nn.tanh)
hidden_layer_1_ = add_layer(hidden_layer_0_, hidden_num,hidden_num,activation_function=tf.nn.tanh)
output_layer_out_ = add_layer(hidden_layer_1_, hidden_num,output_num,activation_function=tf.nn.softmax)


#define loss function:cross entropy
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output_layer_out_), reduction_indices=[1]))

#define how to update network
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

with tf.name_scope('Evaluate'):
    probability = tf.argmax(output_layer_out_,1)

### here start to process data;
input_file_name = "/Users/lqy/Documents/DataSet/f.e.align.train.gz"
src_vocab_,trg_vocab_ = du.create_vocabulary(input_file_name,4000,2000)

word_pairs,word_pairs_id = du.partition_file(input_file_name,src_vocab_,trg_vocab_)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_ = du.get_batch(10, word_pairs_id)
    for i in range(10):
        batch_xs = np.array([x[0] for x in batch_],dtype=np.int32).reshape(len(batch_))
        target_list = [x[1] for x in batch_]
        batch_ys = set_output_gama(target_list,output_num)
        print("before update, the cross_entropy:")
        print(sess.run(cross_entropy, feed_dict={xs:batch_xs,ys:batch_ys}))
        print(sess.run(train_step, feed_dict={xs:batch_xs,ys:batch_ys}))
        print("after update, the cross_entropy:")
        print(sess.run(cross_entropy, feed_dict={xs:batch_xs,ys:batch_ys}))
        print ('finish one batch')
        #print(sess.run(output_layer_out_, feed_dict={xs:batch_xs}))
        #print ("finish one batch!")
        '''
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i % 50 == 0:
            print (compute_accuracy(mnist.test.images,mnist.test.labels))
'''
