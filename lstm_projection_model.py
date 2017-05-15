import tensorflow as tf
import numpy as np
import data_util as du
import os

# THIS IS an ongoing version on lstm training
SRC_VOCAB_SIZE = 40678
WORD_PROJ_SIZE = 100
LSTM_OUT_SIZE_ONE_WORD = 500
TIME_STEP = 3 #in machine translation, it should be the size of a sentence
FIRST_HIDDEN_LAYER = 1000
SECOND_HIDDEN_LAYER = 500
TRG_VOCAB_SIZE = 2000
EPOCH_NUM = 20
LEXICON_MODEL_SAVE_PATH = "/Users/lqy/Documents/lexicon_model_0515"

## inputs = batch_size * time_step * input_size
def projection_LSTM(inputs,n_hidden_units):
    ## construct basic cell
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    batch_size = tf.shape(inputs)[0]
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, time_major=False)
    outputs = tf.reshape(outputs,[batch_size,-1])
    return outputs

def projection_Bidirectional_LSTM(inputs,n_hidden_units):
    forward_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    backward_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0, state_is_tuple=True)
    batch_size = tf.shape(inputs)[0]
    forward_init_state_ = forward_cell.zero_state(batch_size, dtype=tf.float32)
    backward_init_state_ = backward_cell.zero_state(batch_size,dtype=tf.float32)

    outputs_,final_state = tf.nn.bidirectional_dynamic_rnn(forward_cell,backward_cell, inputs,initial_state_fw=forward_init_state_, initial_state_bw=backward_init_state_,time_major=False)

    print 'in the projection outputs = ',outputs_
    print 'in the projection outputs.shape = ',outputs_[0].shape
    concate_outputs = outputs_[0] + outputs_[1]
    concate_outputs = tf.reshape(concate_outputs,[batch_size,-1])
    return concate_outputs


def projection_Linear(inputs,in_size,out_size):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    Biases = tf.Variable(tf.random_normal([1,out_size]))
    outputs = tf.nn.embedding_lookup(Weights, inputs) + Biases
    return outputs

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


## dev_src_sent and dev_trg_sent is consist of word index,
def compute_sent_perplexity(sess, dev_src_sent,dev_trg_sent):
    global outputs_
    global time_s
    dev_src_sent_array = np.array(dev_src_sent)
    time_s[0] = len(dev_src_sent)

    ##print 'in compute_sent_perplexity, dev_src_sent.size=',len(dev_src_sent),'dev_trg_sent.size=',len(dev_trg_sent)

    probs_ = sess.run(outputs_,feed_dict={xs:dev_src_sent_array,time_steps_v:time_s})

    forward_table = np.zeros(len(dev_src_sent) * len(dev_trg_sent)).reshape(len(dev_trg_sent),len(dev_src_sent))

    forward_table.astype(float)

    for trg_word_position_id in range(0,len(dev_trg_sent)):
        forward_table[trg_word_position_id,:] = probs_[:,dev_trg_sent[trg_word_position_id]]

    #print 'the probability table is :',forward_table

    for trg_word_position_id in range(1,len(dev_trg_sent)):
        #tmp = [np.sum(forward_table[trg_word_position_id-1] * (1./len(dev_src_sent)))] * len(dev_src_sent)
        tmp = [np.sum(forward_table[trg_word_position_id-1]) ] * len(dev_src_sent)
        tmp = np.array(tmp, dtype=np.float64)
        ##print 'tmp shape = ',tmp.shape
        ##print 'forward_table[trg_word_position_id,:] shape = ',forward_table[trg_word_position_id,:].shape
        forward_table[trg_word_position_id,:] = forward_table[trg_word_position_id,:] * tmp

    #print 'the sent probability distribution over fora:',forward_table
    sent_prob = np.sum(forward_table[len(dev_trg_sent)-1])
    return sent_prob
    ## START TO COMPUTE THE SENT PROB

## dev_src_sent and dev_trg_sent is consist of word index,
def compute_perplexity(sess, dev_data):
    sum_sent_prob = 0
    for src_sent,trg_sent in dev_data:
        sum_sent_prob += compute_sent_perplexity(sess, src_sent,trg_sent)
    return sum_sent_prob


#xs = tf.placeholder(tf.float32, [None, 3, 100])
## START TO BUILD NETWORK MODEL
xs = tf.placeholder(tf.int32,[None])
time_steps_v = tf.placeholder(tf.int32,[None])

projection_output_linear = projection_Linear(xs,SRC_VOCAB_SIZE,WORD_PROJ_SIZE)
projection_output_linear = tf.reshape(projection_output_linear,[-1,time_steps_v[0],WORD_PROJ_SIZE])
projection_output_lstm = projection_LSTM(projection_output_linear,LSTM_OUT_SIZE_ONE_WORD)
projection_output_lstm = tf.reshape(projection_output_lstm,[-1,LSTM_OUT_SIZE_ONE_WORD])
first_hidden_output = add_layer(projection_output_lstm,LSTM_OUT_SIZE_ONE_WORD,FIRST_HIDDEN_LAYER,activation_function=tf.nn.tanh)
second_hidden_output = add_layer(first_hidden_output,FIRST_HIDDEN_LAYER,SECOND_HIDDEN_LAYER,activation_function=tf.nn.tanh)
outputs_ = add_layer(second_hidden_output,SECOND_HIDDEN_LAYER,TRG_VOCAB_SIZE,activation_function=tf.nn.softmax)

## DEFINE THE LOSS FUNCTION
ys = tf.placeholder(tf.float32,[None,TRG_VOCAB_SIZE])
loss = tf.reduce_mean(tf.square(ys - outputs_))

## DEFINE THE GRADIENT UPDATE APPROACHES
train_step_square_error = tf.train.GradientDescentOptimizer(0.0125).minimize(loss)


## PREPARE TRAINING DATA #####

## START TO LOAD IBM1 LEXICON MODEL
lexicon_prob = du.load_IBM_prob('/Users/lqy/Documents/DataSet/ibm1.actual.ti.final')

## START TO LOAD TRAIN SENTENCEES
with open("/Users/lqy/Documents/DataSet/source_noindex") as f:
    src_sents = f.readlines()

with open("/Users/lqy/Documents/DataSet/target_noindex") as f:
    trg_sents = f.readlines()

## BUILD SENTENCE PAIR (SOURCE SENTENCE -> TARGET SENTENCE)
data_ = [(src_sents[i].strip(),trg_sents[i].strip()) for i in range(len(src_sents))]

## LOAD SOURCE AND TARGET VOCABULARY
srcVocab = du.create_vocabulary("/Users/lqy/Documents/DataSet/source_noindex", SRC_VOCAB_SIZE)
trgVocab = du.create_vocabulary("/Users/lqy/Documents/DataSet/target_noindex", TRG_VOCAB_SIZE)

## train_word_pairs = [(src_sent, trg_word, gama_for_one_trg_word)]
train_word_pairs=[]
count = 0
for src_sent, trg_sent in data_:
    src_words = src_sent.strip().split(' ')
    trg_words = trg_sent.strip().split(' ')
    _,train_word_pairs = du.compute_gama_from_model_lstm(src_words, trg_words, lexicon_prob,train_word_pairs, srcVocab, trgVocab)
    count = count+1
    if count == 20:
        break

## PREPARE DEV SET for computing the perplexity
dev_data_ = []
count = 0
for src_sent,trg_sent in data_:
    src_words = src_sent.strip().split(' ')
    trg_words = trg_sent.strip().split(' ')
    for id, word in enumerate(src_words):
        if not(word in srcVocab):
            src_words[id] = "<unk>"
    for id, word in enumerate(trg_words):
        if not(word in trgVocab):
            trg_words[id] = "<unk>"
    dev_data_.append(([srcVocab[word] for word in src_words],[trgVocab[word] for word in trg_words]))

    count = count +1
    if count == 2:
        break

time_s = np.random.randint(1,3,1)


print 'START TO TRAIN'

HAS_MODEL = os.path.isfile(LEXICON_MODEL_SAVE_PATH)

with tf.Session() as sess:
    if HAS_MODEL == False:
        sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if HAS_MODEL == True:
        saver.restore(sess,LEXICON_MODEL_SAVE_PATH)
    for _ in range(EPOCH_NUM):
        from time import clock
        perplexity_value = compute_perplexity(sess,dev_data_)
        print 'perplexity value is equal to ',perplexity_value

        start = clock()
        for src_sent,trg_word,gama_line in train_word_pairs:
            src_words_input = np.array(src_sent,dtype = np.int32)
            gama = np.zeros(TRG_VOCAB_SIZE * len(gama_line)).reshape(len(gama_line),TRG_VOCAB_SIZE)
            gama[:,trg_word] = gama_line
            time_s[0] = len(gama_line)
            #print 'loss is = ',sess.run(loss,feed_dict={xs:src_words_input,time_steps_v:time_s,ys:gama})
            sess.run(train_step_square_error,feed_dict={xs:src_words_input,time_steps_v:time_s,ys:gama})
            #print 'after one update loss is = ',sess.run(loss,feed_dict={xs:src_words_input,time_steps_v:time_s,ys:gama})
        perplexity_value = compute_perplexity(sess,dev_data_)
        print 'after training, the perplexity value is equal to ', perplexity_value
        end = clock()
    print "each batch cost time =",(end - start) * (1./1000.)
    print 'start to save model'
    saver.save(sess,LEXICON_MODEL_SAVE_PATH)
    print 'finish model saving'



'''print 'size of train_word_pairs = ',len(train_word_pairs)

for src,trg_word,gama_line in train_word_pairs:
    print '##############'
    print 'src_sent = ',src
    print 'trg_word = ',trg_word
    print 'accordingly gama list:',gama_line
    print 'gama list length = ',len(gama_line)
    print '##############'d

for src_sent,trg_word,gama_line in train_word_pairs:
    print 'for trg_word = ',trg_word
    print 'len of gama_line = ', len(gama_line)
    src_words_input = np.array(src_sent)
    gama = np.zeros(TRG_VOCAB_SIZE * len(gama_line)).reshape([len(gama_line),TRG_VOCAB_SIZE])
    print 'gama shape = ',gama.shape
    gama[:,trg_word] = gama_line
    print 'the gama list should be:', gama[:,trg_word]
'''
