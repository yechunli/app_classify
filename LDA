import pandas as pd
import numpy as np
import jieba.analyse
import collections
from gensim import corpora, models
from sklearn.feature_extraction.text import HashingVectorizer
import operator
from sklearn.preprocessing import OneHotEncoder

data_file = 'F:\\标注比赛\\大数据应用分类标注-选手\\apptype_train.dat'
label_id = 'F:\\标注比赛\\大数据应用分类标注-选手\\apptype_id_name.txt'

stop_word = [] 
with open('F:\标注比赛\stop_words.txt') as f: 
    line = f.readline() 
    while line: 
        line = line.split('\n') 
        stop_word.append(line[0]) 
        line = f.readline() 
    stop_word = set(stop_word)

def data_process(data_file, model='test'): 
    enc = OneHotEncoder()
    i = -1  
    with open(data_file, encoding='utf-8') as f: 
        line = f.readline() 
        all_words = [] 
        sentenses = [] 
        app_id_list = [] 
        length = []
        data = []
        data_sentense = []
        while line: 
            i = i + 1
            #if i in youxi:
            #    line = f.readline()
            #    continue
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
            word_line = list(jieba.analyse.extract_tags(app_desc, withWeight=False, allowPOS=('n', 'vn')))
            #word_line = list(jieba.cut(app_desc))
            for word in word_line:
                if word not in stop_word:#and word not in ['qingkan520','www','com','http']:
                    desc_words.append(word)
                    all_words.append(word)
            #if app_id == 21:
            #    print(line)
            #    print(desc_words)
            #    break
            line = f.readline()

            sentenses.append(desc_words)
            
            #i = i+ 1
            #if i == 1000:
            #    break
#        counter = collections.Counter(all_words)
#        word_dict = counter.most_common(30000)
#        dict_list = [x[0] for x in word_dict]
#        dictionary = {}
#        for i in range(len(dict_list)):
#            dictionary[dict_list[i]] = i

        
        num_doc_train = len(sentenses)
        
#        for queue in sentenses:
#            for i in range(len(queue)):
#                data_tmp = dictionary.get(queue[i])

#                if not data_tmp:
#                    continue
#                data_sentense.append(data_tmp)
#            length.append(len(data_sentense))
#            data.append(data_sentense)
#        data_len = np.mean(length)
        
#        for i in range(num_doc_train):
#            if length[i] >= data_len:
#                data[i] = data[i][:data_len]
#            else:
#                data[i] = data[i]+[0 for x in range(data_len-length[i])]
#    label = enc.fit_transform(np.array(app_id_list).reshape(-1, 1)).toarray()
#    return data, data_len, label
        dictionary = corpora.Dictionary(sentenses)

        dictionary.filter_extremes(no_below=100, no_above=0.2, keep_n=2000)
        dictionary.compactify()

        corpus = [dictionary.doc2bow(test) for test in sentenses]

        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        return corpus, dictionary, num_doc_train, app_id_list
        
def LDA_model(corpus_tfidf, dictionary, num_doc_train):
    lda_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2, chunksize=100, passes=2, iterations=5,
                                minimum_probability=0.01, random_state=1)
    topic = []
    for i in range(num_doc_train):
        topic_id = lda_model[corpus_tfidf[i]]
        topic_id.sort(key=operator.itemgetter(1))
        topic.append(topic_id[-1][0])
    print(lda_model.print_topics())
    print(lda_model.log_perplexity(corpus_tfidf))
    return topic
    
def label_compare(topic, app_id_list):
    #print(sorted(app_id_list))
    return
    
corpus_tfidf, dictionary, num_doc_train, app_id_list = data_process(data_file, 'train')
topic = LDA_model(corpus_tfidf, dictionary, num_doc_train)
label_compare(topic, app_id_list)

