from cgi import test
import json
from math import log2
import re
import feather
from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
import gensim
import numpy as np
import heapq
def get_api_desc_json(df):
    api_desc_json={}
    for i in range(df.shape[0]):
        api_desc_json[df.loc[i].api]=df.loc[i].description
    return api_desc_json

def get_train_ques_idf(data_json):
    sents=[]
    for item in data_json:
        sents.append(' '.join(item[0]))
    sents = [word_tokenize(sent) for sent in sents]
    print('sents_load')
    corpus = TextCollection(sents)
    print('IDF load')
    corpus_json={}
    word_json = json.load(open('./deepAPI_python_input/desc_2.json'))
    i=0
    for item in word_json:
        if item != '<pad>' or item != '<s>' or item != '</s>' or item != '<unk>':
            corpus_json[item] = corpus.idf(item)
            i+=1
            if i%500==0:
                print(item)

    
    return corpus_json  
def get_cos_sim(v1,v2):
    num = float(np.dot(v1,v2))
    denom = np.linalg.norm(v1)*np.linalg.norm(v2)
    return 0.5+0.5*(num/denom) if denom != 0 else 0
def get_corpus_idf(word, corpus):
    if word in corpus:
        return corpus[word]
    else:
        return 0
def sim_two_sen(sen1, sen2, corpus):
    total_idf = 0.0
    total_sim = 0.0
    for word1 in sen1:
        if word1 in w2v.wv.vocab:
            current_sim=-10.0
            for word2 in sen2:
                if word2 in w2v.wv.vocab:
                    cur_cal = get_cos_sim(w2v.wv[word1],w2v.wv[word2])
                    if cur_cal>current_sim:
                        current_sim = cur_cal
            total_sim+=current_sim*get_corpus_idf(word1,corpus)
            total_idf+=get_corpus_idf(word1,corpus)

            
    return total_sim/total_idf if total_idf !=0 else 0
def total_sim_two_sen(sen1,sen2,corpus):
    return 2*sim_two_sen(sen1,sen2,corpus)*sim_two_sen(sen2,sen1,corpus)/(sim_two_sen(sen1,sen2,corpus)+sim_two_sen(sen2,sen1,corpus)) if (sim_two_sen(sen1,sen2,corpus)+sim_two_sen(sen2,sen1,corpus)) != 0 else 0
def get_top_sim_questions(query, data_json,corpus,top_k):
    sim_list=[total_sim_two_sen(query, item[0], corpus) for item in data_json]
    # sim_list=[]
    # for item in data_json:
    #     sim_list.append(total_sim_two_sen(query,item[0],corpus))
    re1 = list(heapq.nlargest(top_k, range(len(sim_list)), sim_list.__getitem__))
    top_question_list=[data_json[i] for i in re1]
    sim_list = [sim_list[i] for i in re1]
    return top_question_list, sim_list

def recommend_top(query,top_question_list, sim_list_from,api_desc_json, corpus,top_k):
    API_list = [item[2] for item in top_question_list]
    sim_list=[]
    for item in top_question_list:
        curr_sim_so=[]
        curr_sim_desc=[]
        for API in item[2]:
            API_list_id=get_API_list_id(API,API_list)
            sum_APISO_Query = (np.sum((np.array([sim_list_from[i] for i in API_list_id])))*log2(len(API_list_id))/len(API_list_id)) if len(API_list_id)!=0 else 0
            #sim_API_SO = np.min(1.0,sum_APISO_Query)
            sim_API_SO = sum_APISO_Query if sum_APISO_Query < 1 else 1.0
            sim_API_desc = get_desc_query_sim(API,query,api_desc_json,corpus)
            curr_sim_desc.append(sim_API_desc)
            curr_sim_so.append(sim_API_SO)
            #sim_final = 2*sim_API_SO*sim_API_desc/(sim_API_SO+sim_API_desc) if (sim_API_SO+sim_API_desc)!=0 else 0
            #curr_sim.append(sim_final)
        sim_final = 2*np.max(np.array(curr_sim_so))*np.max(np.array(curr_sim_desc))/(np.max(np.array(curr_sim_so))+np.max(np.array(curr_sim_desc))) if (np.max(np.array(curr_sim_so))+np.max(np.array(curr_sim_desc)))!=0 else 0
        sim_list.append(np.mean(np.array(sim_final)))
    re1 = list(heapq.nlargest(top_k, range(len(sim_list)), sim_list.__getitem__))
    result_APIs = [top_question_list[i][2] for i in re1]
    return result_APIs
    
    

def get_API_list_id(API, API_list):
    API_list_id=[i for i in range(len(API_list)) if API in API_list[i]]
    return API_list_id
def get_desc_query_sim(API,query,api_desc_json,corpus):
    return total_sim_two_sen(api_desc_json[API].split(' '),query, corpus) if API in api_desc_json else 0

if __name__ == '__main__':
    all_data_json = json.load(open('./data/all_data_list.json'))
    #all_api_desc_df = feather.read_dataframe('./code/api.feather')
    print('read over')
    #corpus = get_train_ques_idf(all_data_json[0:len(all_data_json)-10000])
    corpus = json.load(open('./data/corpus.json'))
    print('IDF_load')
    w2v = gensim.models.Word2Vec.load('./BIKER-python/w2v_model_stemmed')
    print('w2v load') 
    api_desc_json=json.load(open('./data/api_desc.json'))
    # with open('./data/api_desc.json','w') as f:
    #     json.dump(api_desc_json,f)
    # with open('./data/corpus.json','w') as f:
    #     json.dump(corpus,f)
    #top_question, sim_list = get_top_sim_questions(all_data_json[-1][0],all_data_json[0:len(all_data_json)-10000],corpus,50)
    train_data_json = all_data_json[0:len(all_data_json)-10000]
    test_data_json = all_data_json[len(all_data_json)-10000:-1]
    with open('./data/BIKER_result.txt','w') as f:
        for iii in range(len(test_data_json)):
            f.write(' '.join(test_data_json[iii][2])+'\n')
            query = test_data_json[iii][0]
            top_question, sim_list = get_top_sim_questions(query,train_data_json,corpus,50)
            return_APIs = recommend_top(query, top_question, sim_list, api_desc_json, corpus, 10)
            for item in return_APIs:
                f.write(' '.join(item)+'\n')
            if iii%50==0:
                print(iii)

    # print(all_data_json[-1])
    # print('.......................')
    # for item in list_id:
    #     print(all_data_json[item])
    #     print('******')




