import sys
import gzip
import random
import numpy as np

reload(sys)

sys.setdefaultencoding('utf8')


def partition_file(in_file):
    word_pairs = []
    with gzip.open(in_file) as f:
        lines = f.readlines()
        for line in lines:
            src_sent,trg_sent,alignment = line.split('#')
            src_words = src_sent.strip().split(' ')
            trg_words = trg_sent.strip().split(' ')
            align_list = alignment.strip().split(' ')
            align_list = align_list[1:]
            align = []
            for i in range(1,len(align_list)-1,3):
                align.append((int(align_list[i]),int(align_list[i+1])))

            for src_position, trg_position in align:
                word_pairs.append((src_words[src_position],trg_words[trg_position]))
                #print src_position, '->', trg_position
    return word_pairs

## get train data from tama's file,prepare a tuple
def prepare_train_data(in_file):
    word_pairs = []
    with gzip.open(in_file) as f:
        lines = f.readlines()
        for line in lines:
            src_sent,trg_sent,alignment = line.split('#')
            src_words = src_sent.strip().split(' ')
            trg_words = trg_sent.strip().split(' ')
            align_list = alignment.strip().split(' ')
            align_list = align_list[1:]
            align = []
            for i in range(1,len(align_list)-1,3):
                align.append((int(align_list[i]),int(align_list[i+1])))

            for src_position, trg_position in align:
                a =['','','']
                if src_position -1 <0:
                    a[0]="<sb>"
                else:
                    a[0] = src_words[src_position-1]

                a[1] = src_words[src_position]

                if src_position + 1 >= len(src_words):
                    a[2] = "<sb>"
                else:
                    a[2] = src_words[src_position+1]

                word_pairs.append(((a[0],a[1],a[2]),trg_words[trg_position]))
        return word_pairs


def get_batch(batch_size,word_pairs):
    word_pair_batch_ = random.sample(word_pairs,batch_size)
    return word_pair_batch_


#this is for triple data, from tama's training data
def create_vocabulary(in_file,src_vocab_size,trg_vocab_size):
    src_vocab_ = {}
    trg_vocab_ = {}
    with gzip.open(in_file) as f:
        lines = f.readlines()
        src_sents = []
        trg_sents = []
        for line in lines:
            src_sent,trg_sent,alignment = line.split('#')
            src_sents.append(src_sent.strip())
            trg_sents.append(trg_sent.strip())
#start to process the vocabualry
        print 'start to prepare vocabulary data for source and target'
        for src_sent in src_sents:
            for word in src_sent.strip().split(' '):
                if src_vocab_.has_key(word):
                    src_vocab_[word] += 1
                else:
                    src_vocab_[word] = 1
        #src_vocab_list_ = sorted(src_vocab_.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)
        #print 'finish loading src data',len(src_vocab_)
        src_vocab_list_=sorted(src_vocab_,key=src_vocab_.get,reverse=True)
        if len(src_vocab_list_) > src_vocab_size:
            src_vocab_ = dict( (y,x) for (x,y) in enumerate((["<unk>"] +src_vocab_list_[:(src_vocab_size-1)])))

        for trg_sent in trg_sents:
            for word in trg_sent.strip().split(' '):
                if trg_vocab_.has_key(word):
                    trg_vocab_[word] +=1
                else:
                    trg_vocab_[word] = 1
        #print 'finish loading trg data', len(trg_vocab_)
        trg_vocab_list_=sorted(trg_vocab_,key=trg_vocab_.get,reverse=True)
        if len(trg_vocab_list_) > trg_vocab_size:
            trg_vocab_ = dict((y,x) for (x,y) in enumerate((["<unk>"] +trg_vocab_list_[:trg_vocab_size-1])))

        return src_vocab_,trg_vocab_
        #print len(src_vocab_)
        #print len(trg_vocab_)

def create_vocabulary(in_file,vocab_size):
    vocab_ = {}
    with open(in_file) as f:
        lines = f.readlines()
        sents = []
        for line in lines:
            sent = line.strip()
            sents.append(sent)
#start to process the vocabualry
        print 'start to prepare vocabulary data for source and target'
        for sent in sents:
            for word in sent.strip().split(' '):
                if vocab_.has_key(word):
                    vocab_[word] +=1
                else:
                    vocab_[word] = 1
        #src_vocab_list_ = sorted(src_vocab_.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)
        #print 'finish loading src data',len(src_vocab_)
        vocab_list_=sorted(vocab_,key=vocab_.get,reverse=True)
        if len(vocab_list_) > vocab_size:
            vocab_ = dict( (y,x) for (x,y) in enumerate((["<unk>"] +vocab_list_[:(vocab_size-1)])))
        else:
            vocab_ = dict( (y,x) for (x,y) in enumerate((["<unk>"] +vocab_list_[:])))
        print len(vocab_list_)
        return vocab_

def partition_file(in_file,srcVocab,trgVocab):
    word_pairs = []
    word_pairs_id =[]
    with gzip.open(in_file) as f:
        lines = f.readlines()
        for line in lines:
            src_sent,trg_sent,alignment = line.split('#')
            src_words = src_sent.strip().split(' ')
            trg_words = trg_sent.strip().split(' ')
            align_list = alignment.strip().split(' ')
            align_list = align_list[1:]
            align = []
            for i in range(1,len(align_list)-1,3):
                align.append((int(align_list[i]),int(align_list[i+1])))

            for src_position, trg_position in align:
                src_word = src_words[src_position] if src_words[src_position] in srcVocab else "<unk>"
                trg_word = trg_words[trg_position] if trg_words[trg_position] in trgVocab else "<unk>"
                word_pairs.append((src_word,trg_word))
                word_pairs_id.append((srcVocab[src_word], trgVocab[trg_word]))


                #print src_position, '->', trg_position
    return word_pairs, word_pairs_id

def compute_gama_from_model(src_sent, trg_sent, lexicon_probs,train_word_pairs, srcVocab, trgVocab, alignment_probs=None):
    prob_table = np.zeros(len(src_sent) * len(trg_sent)).reshape(len(trg_sent),len(src_sent))
    for i in range(0,len(trg_sent)):
        for j in range(0,len(src_sent)):
            if((src_sent[j],trg_sent[i]) in lexicon_probs):
                prob_table[i][j] = lexicon_probs[(src_sent[j],trg_sent[i])]
            else:
                prob_table[i][j] = 0

    forward_table = np.copy(prob_table)
    backward_table = np.copy(prob_table)

    #compute forward table
    for trg_index in range(1,len(trg_sent)):
        tmp = [np.sum(forward_table[trg_index-1] * (1./len(src_sent)))] * len(src_sent)
        tmp = np.array(tmp, dtype=np.float64)
        forward_table[trg_index] = forward_table[trg_index] * tmp

    #compute backward table
    for trg_index in range(len(trg_sent)-2, -1, -1):
        tmp = [np.sum(backward_table[trg_index+1] * (1./len(src_sent)))] * len(src_sent)
        tmp = np.array(tmp,dtype=np.float64)
        backward_table[trg_index] = backward_table[trg_index] * tmp

    gama_table = forward_table * backward_table

    for row in range((gama_table.shape)[0]):
        try:
            if np.sum(gama_table[row]) == 0:
                '''
                print 'we are facing 0 in gama_table[row], row = ',row
                print 'before gama table '
                print gama_table[row]'''
                #gama_table[row] = gama_table[row] * (1./len(src_sent))
                gama_table[row] = gama_table[row] + (1./len(src_sent))
                print 'gama problem!!'
            else:
                gama_table[row] = gama_table[row] / np.sum(gama_table[row])
        except :
            print row

    for i_word, target_word in enumerate(trg_sent):
        for j, src_word in enumerate(src_sent):
            if src_word not in srcVocab:
                src_word = 'wir-----------'
            if target_word not in trgVocab:
                target_word = 'we---------------'
            train_word_pairs.append((srcVocab[src_word], trgVocab[target_word], gama_table[i_word][j]))


    return gama_table, train_word_pairs


def compute_gama_from_model_lstm(src_sent, trg_sent, lexicon_probs,train_word_pairs, srcVocab, trgVocab, alignment_probs=None):
    prob_table = np.zeros(len(src_sent) * len(trg_sent)).reshape(len(trg_sent),len(src_sent))
    for i in range(0,len(trg_sent)):
        for j in range(0,len(src_sent)):
            if((src_sent[j],trg_sent[i]) in lexicon_probs):
                prob_table[i][j] = lexicon_probs[(src_sent[j],trg_sent[i])]
            else:
                prob_table[i][j] = 0

    forward_table = np.copy(prob_table)
    backward_table = np.copy(prob_table)

    #compute forward table
    for trg_index in range(1,len(trg_sent)):
        tmp = [np.sum(forward_table[trg_index-1] * (1./len(src_sent)))] * len(src_sent)
        tmp = np.array(tmp, dtype=np.float64)
        forward_table[trg_index] = forward_table[trg_index] * tmp

    #compute backward table
    for trg_index in range(len(trg_sent)-2, -1, -1):
        tmp = [np.sum(backward_table[trg_index+1] * (1./len(src_sent)))] * len(src_sent)
        tmp = np.array(tmp,dtype=np.float64)
        backward_table[trg_index] = backward_table[trg_index] * tmp

    gama_table = forward_table * backward_table

    for row in range((gama_table.shape)[0]):
        try:
            if np.sum(gama_table[row]) == 0:
                '''
                print 'we are facing 0 in gama_table[row], row = ',row
                print 'before gama table '
                print gama_table[row]'''
                #gama_table[row] = gama_table[row] * (1./len(src_sent))
                gama_table[row] = gama_table[row] + (1./len(src_sent))
                print 'gama problem!!'
            else:
                gama_table[row] = gama_table[row] / np.sum(gama_table[row])
        except :
            print row

    for i_word, target_word in enumerate(trg_sent):
        if not (target_word in trgVocab):
            target_word = "<unk>"
        for word_no, src_word in enumerate(src_sent):
            if not(src_sent[word_no] in srcVocab):
                src_sent[word_no] = "<unk>"
                print 'there is unknown word, = ',src_sent[word_no]
            #src_sent[word_no] = srcVocab[src_sent[word_no]]
        #train_word_pairs.append((src_sent, target_word, gama_table[i_word][:]))
        train_word_pairs.append(([srcVocab[word] for word in src_sent], trgVocab[target_word], gama_table[i_word][:]))

    return gama_table, train_word_pairs


def load_IBM_prob(lexicon_file):
    with open(lexicon_file) as f:
        lines = f.readlines()
        lexicon_prob = {}
        for line in lines:
            src_word,trg_word, prob = line.strip().split(' ')
            lexicon_prob[(src_word,trg_word)]= float(prob)

        print 'finish~', len(lexicon_prob)
        return lexicon_prob


'''
lexicon_prob = load_IBM_prob('/Users/lqy/Documents/DataSet/ibm1.actual.ti.final')

with open("/Users/lqy/Documents/DataSet/source_noindex") as f:
    src_sents = f.readlines()

with open("/Users/lqy/Documents/DataSet/target_noindex") as f:
    trg_sents = f.readlines()

data_ = [(src_sents[i],trg_sents[i]) for i in range(len(src_sents))]
srcVocab = create_vocabulary("/Users/lqy/Documents/DataSet/source_noindex", 40678)
trgVocab = create_vocabulary("/Users/lqy/Documents/DataSet/target_noindex", 2000)
train_word_pairs=[]
count = 0
print 'length of data : ',len(data_)
for src_sent, trg_sent in data_:
    src_words = src_sent.strip().split(' ')
    trg_words = trg_sent.strip().split(' ')
    _,train_word_pairs = compute_gama_from_model(src_words, trg_words, lexicon_prob,train_word_pairs, srcVocab, trgVocab)
    count +=1
    if count % 1000 == 0:
        print 'finish sentence = ', count
    if count == 60000:
        break

print len(train_word_pairs)
print 'count is = ', count
print 'finish'

'''

'''
train_data_ = prepare_train_data('/Users/lqy/Documents/DataSet/f.e.align.train.gz')

print len(train_data_)

for i in range(20):
    print train_data_[i]
'''
