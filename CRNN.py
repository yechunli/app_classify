import pandas as pd
import numpy as np
import jieba.analyse
import collections
from gensim import corpora, models
from sklearn.feature_extraction.text import HashingVectorizer
import operator
from sklearn.preprocessing import OneHotEncoder

data_file = 'F:\\label_competition\\大数据应用分类标注-选手\\apptype_train.dat'
label_id = 'F:\\label_competition\\大数据应用分类标注-选手\\apptype_id_name.txt'

#TextCNN
import tensorflow as tf


class TextRCNN:
    def __init__(self, sequence_length, num_classes, vocab_size, word_embedding_size, context_embedding_size,
                 cell_type, hidden_size, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_text)

        # Embeddings
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, word_embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Bidirectional(Left&Right) Recurrent Structure
        with tf.name_scope("bi-rnn"):
            fw_cell = self._get_cell(context_embedding_size, cell_type)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = self._get_cell(context_embedding_size, cell_type)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedded_chars,
                                                                                       sequence_length=text_length,
                                                                                       dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name="context_left")
            self.c_right = tf.concat([self.output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word-representation"):
            self.x = tf.concat([self.c_left, self.embedded_chars, self.c_right], axis=2, name="x")
            embedding_size = 2*context_embedding_size + word_embedding_size

        with tf.name_scope("text-representation"):
            W2 = tf.Variable(tf.random_uniform([embedding_size, hidden_size], -1.0, 1.0), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b2")
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, W2) + b2)

        with tf.name_scope("max-pooling"):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        with tf.name_scope("output"):
            W4 = tf.get_variable("W4", shape=[hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")
            l2_loss += tf.nn.l2_loss(W4)
            l2_loss += tf.nn.l2_loss(b4)
            self.logits = tf.nn.xw_plus_b(self.y3, W4, b4, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)
        
stop_word = [] 
with open('F:\\label_competition\\stop_words.txt') as f: 
    line = f.readline() 
    while line: 
        line = line.split('\n') 
        stop_word.append(line[0]) 
        line = f.readline() 
    stop_word = set(stop_word)

def word_read(data_file, model='test'): 
    enc = OneHotEncoder()
    i = -1  
    with open(data_file, encoding='utf-8') as f: 
        line = f.readline() 
        all_words = [] 
        sentenses = [] 
        app_id_list = [] 

        while line: 
            i = i + 1
            desc_words = []
            word_list = []
            line = line.split('\n')
            line = line[0].split('\t')
            if model == 'train':   
                two_id = line[1].split('|')
                app_id = (int(two_id[0]) // 100)
                #if app_id != 1401 and app_id != 1409:
                #    line = f.readline()
                #    continue
                app_id_list.append(app_id)
            app_desc = line[-1]
            word_line = list(jieba.cut(app_desc))
            for word in word_line:
                if word not in stop_word:#and word not in ['qingkan520','www','com','http']:
                    desc_words.append(word)
                    all_words.append(word)
            line = f.readline()
            sentenses.append(desc_words)
            
        counter = collections.Counter(all_words)
        word_dict = counter.most_common(30000)
        dict_list = [x[0] for x in word_dict]
        
        label = enc.fit_transform(np.array(app_id_list).reshape(-1, 1)).toarray()
        
        return dict_list, sentenses, label
        
def word_to_num(dict_list, sentenses):
    length = []
    data = []

    dictionary = {}
    for i in range(len(dict_list)):
        dictionary[dict_list[i]] = i

    for queue in sentenses:
        data_sentense = []
        for i in range(len(queue)):
            data_tmp = dictionary.get(queue[i])
            if not data_tmp:
                continue
            data_sentense.append(data_tmp)
        length.append(len(data_sentense))
        data.append(data_sentense)
    data_len = np.mean(length)
    
    return data, int(data_len), length
    
def data_process(data, data_len, length):
    for i in range(len(data)):
        if length[i] >= data_len:
            data[i] = data[i][:data_len]
        else:
            data[i] = data[i]+[0 for x in range(data_len-length[i])]
    return data
    
#step = 'data_prepare'
step = 'model'
if step == 'data_prepare':
    dict_list, sentenses, label = word_read(data_file, 'train')
    data, data_len, length = word_to_num(dict_list, sentenses)
    data_input = data_process(data, data_len, length)
    pd.DataFrame(data_input).to_csv('F:\\label_competition\\data.csv', index=False)
    pd.DataFrame(label).to_csv('F:\\label_competition\\label.csv', index=False)
if step == 'model':
    data_input = pd.read_csv('F:\\label_competition\\data.csv')
    label = pd.read_csv('F:\\label_competition\\label.csv')
    data_len = len(data_input.columns)

    iterator = 10000
    num_classes = 26
    embedding_size = 200
    vocab_size = 30000
    context_embedding_size = 300
    hidden_size = 1000
    rcnn = TextRCNN(data_len, num_classes, vocab_size, embedding_size, context_embedding_size,
                     'lstm', hidden_size, l2_reg_lambda=0.01)

    num_sample = len(data_input)
    batch_size = 100
    step = int(num_sample // batch_size)
    local_init = tf.local_variables_initializer()
    global_init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(local_init)
        for i in range(iterator):
            for j in range(step):
                feed_dict = {rcnn.input_text:data_input[j*step:(j+1)*step], rcnn.input_y:label[j*step:(j+1)*step], rcnn.dropout_keep_prob:0.8}
                y3 = sess.run(rcnn.logits, feed_dict=feed_dict)
                sess.run(rcnn.train_op, feed_dict=feed_dict)
            if i%10 == 0:
                loss = sess.run(rcnn.loss, feed_dict=feed_dict)
                print('loss:', loss)
