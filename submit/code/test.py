# encoding: utf-8
'''
Created on Oct 26, 2019

@author: Yongrui Huang
'''

# coding=utf8
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pickle
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import plotly.graph_objs as go
import plotly.offline as py

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import jieba.posseg as pseg
import jieba.analyse
import re
import jieba

import lightgbm as lgb

plt.style.use('seaborn')

train_data = pd.read_csv('../data/task3/train.csv')
test_data = pd.read_csv('../data/task3/task3_new_stage2.csv')

all_target = train_data['label']
train_data = train_data.drop(['label'], axis=1)
all_data = train_data.append(test_data, ignore_index=True)
train_idx = all_data.index[:-len(test_data)]
test_idx = all_data.index[-len(test_data):]


def clean_text(news):
    """
    """
    
    remove_chars = '[\t’ !"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', news)

def same_news(news_a, news_b):
    """
    """
    news_a = clean_text(news_a)
    news_b = clean_text(news_b)
    if len(news_a) < 5 or len(news_b) < 5:
        return False
    
    cnt_same = 10
    cnt_same = min([cnt_same, len(news_a), len(news_b)])
    for i in range(cnt_same):
        if news_a[i] != news_b[i]:
            return False
    return True

all_data['clean_text'] = all_data.text.apply(clean_text)
sorted_all_data = all_data.sort_values(by='clean_text')
news_label = 1
news_lables = np.zeros(sorted_all_data.shape[0], dtype=np.int64)
same_news_train_idx = []
for i in range(sorted_all_data.shape[0] - 1):
    news_a = sorted_all_data.iloc[i]['text']
    news_b = sorted_all_data.iloc[i + 1]['text']
    news_lables[i] = news_label
    if same_news(news_a, news_b) == False:
        news_label += 1
    else:
        same_news_train_idx.append(sorted_all_data.iloc[i : i + 1].index[0])
news_lables[-1] = news_label
sorted_all_data['news_label'] = news_lables
#训练集重复news数量
len(same_news_train_idx)

train_data_with_news_label = sorted_all_data.loc[train_idx]
test_data_with_news_label = sorted_all_data.loc[test_idx]
cnt_shown_news = 0
train_same_idx = []
mp_test_id_pre = {}
def find_news_in_train_data(sample):
    """
    """
    global cnt_shown_news, mp_test_id_pre
    news_in_train = train_data_with_news_label[train_data_with_news_label['news_label'] == sample.news_label]
    if len(news_in_train) > 0:
#         print(news_in_train.iloc[0].text)
#         print(sample.text)
#         print(news_in_train.index) 
        mp_test_id_pre[sample['id']] = all_target.loc[news_in_train.iloc[:1].index[0]]
        train_same_idx.extend(list(news_in_train.index))

        cnt_shown_news += 1
        
test_data_with_news_label.apply(find_news_in_train_data, axis=1)
print(cnt_shown_news)
print(len(set(train_same_idx)))
# print(mp_test_id_pre)

news_label_feature = ['news_label', 'forwards_num']
value_counts_news_labels = sorted_all_data.news_label.value_counts()
sorted_all_data['forwards_num'] = sorted_all_data.news_label.apply(lambda x:value_counts_news_labels[x]-1)

all_data['news_label'] = sorted_all_data['news_label']
all_data['forwards_num'] = sorted_all_data['forwards_num']

texts = list(all_data['text'])



jieba_mp_news_location_number = {}
jieba_mp_news_number_location = {}
count_location = 0

def get_jieba_location(sentence):
    """
    """
    global count_location, jieba_mp_news_location_number, jieba_mp_news_number_location
    words = pseg.cut(sentence)
    for word, flag in words:
        if len(word) == 1:
            continue
        if flag == 'ns':
            if word not in jieba_mp_news_location_number:
                count_location += 1
                jieba_mp_news_location_number[word] = count_location
                jieba_mp_news_number_location[count_location] = word
            return jieba_mp_news_location_number[word]
    return -1

jieba_location = all_data.text.apply(get_jieba_location)
print(count_location)
print(jieba_mp_news_location_number)

province_list = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林','黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东',\
                '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏',\
                '新疆', '台湾', '香港', '澳门']
def get_province(x):
    """
    """
    if str(x) == 'nan':
        return x
    if '其他' in x:
        return x
    if '海外' in x:
        return x
    for province in province_list:
        if province in x:
            return province
news_province_location = all_data.text.apply(get_province)

features_news_location = ['jieba_news_location', 'news_province']
all_data['jieba_news_location'] = jieba_location
all_data['news_province'] = pd.factorize(news_province_location)[0]

def word_segment(sentence):
    words = jieba.cut(sentence)
    return ','.join(words).split(',')

stop_words = set()
def load_stopwords():
    """
    """
    with open('stopwords.txt', 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            stop_words.add(line.strip())

load_stopwords()

def remove_stopwords(word_lists):
    """
    """
    res = []
    for word in word_lists:
        if word not in stop_words:
            res.append(word)
    return ' '.join(res)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_tfidf_svd_matrix(texts, n_output):
    """
    """
    corpus = []
    for text in texts:
        words = word_segment(str(text))
        use_words = []
        for word in words:
            if word in stop_words:
                continue
            else:
                use_words.append(word)
        corpus.append(' '.join(use_words))
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(corpus)
    svd = TruncatedSVD(n_components=n_output, n_iter=7, random_state=42)
    tf_idf_svd = svd.fit_transform(tfidf_matrix)
    
    return tf_idf_svd


# all_data['news_segment'] = all_data.text.apply(word_segment)
# all_data['news_segment_rm_stopword'] = all_data.news_segment.apply(remove_stopwords)
news_tf_idf_svd = build_tfidf_svd_matrix(all_data['text'], 100)

tf_idf_news_columns_names = ['td_idf_news_%d' % i for i in range(news_tf_idf_svd.shape[1])]
df_tf_idf_news_svd = pd.DataFrame(news_tf_idf_svd, columns = tf_idf_news_columns_names)
df_tf_idf_news_svd.head()

user_tf_idf_svd = build_tfidf_svd_matrix(all_data['userDescription'], 100)
print(user_tf_idf_svd.shape)

tf_idf_user_columns_names = ['td_idf_user_%d' % i for i in range(user_tf_idf_svd.shape[1])]
df_tf_idf_user_svd = pd.DataFrame(user_tf_idf_svd, columns = tf_idf_user_columns_names)
df_tf_idf_user_svd.head()

from gensim import corpora, models
import jieba.posseg as jp, jieba

def get_lda_model(texts, topic_number):
    """
    """
    words_list = []
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd') 
    for text in news_texts:
        words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stop_words]
        words_list.append(words)
    dictionary = corpora.Dictionary(words_list)
    corpus = [dictionary.doc2bow(words) for words in words_list]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics = topic_number)
    return lda, corpus

news_texts = list(all_data['text'])
lda_news, news_corpus = get_lda_model(news_texts, 20)

for topic in lda_news.print_topics(num_words=5):
    print(topic)
    
lda_matrix_news = lda_news.inference(news_corpus)[0]
print(lda_matrix_news.shape)
lda_news_columns_names = ['lda_news_%d' % i for i in range(lda_matrix_news.shape[1])]
df_lda_news = pd.DataFrame(lda_matrix_news, columns = lda_news_columns_names)
df_lda_news.head()

lda_user, user_corpus = get_lda_model(list(all_data['userDescription']), 10)

for topic in lda_user.print_topics(num_words=5):
    print(topic)
    
lda_matrix_user = lda_user.inference(user_corpus)[0]
lda_matrix_user.shape

lda_user_columns_names = ['lda_user_%d' % i for i in range(lda_matrix_user.shape[1])]
df_lda_user = pd.DataFrame(lda_matrix_user, columns = lda_user_columns_names)
df_lda_user.head()

import json

def read_json_file(file_name):
    with open(file_name, 'r') as f:
        return json.loads(f.read())
mp_pid_TRUE = read_json_file('image_middle_file/mp_pid_TRUE.json')
mp_pid_rumor = read_json_file('image_middle_file/mp_pid_rumor.json')
mp_pid_width = read_json_file('image_middle_file/mp_pid_width.json')
mp_pid_height = read_json_file('image_middle_file/mp_pid_height.json')

image_features = ['width_mean', 'true_scores_mean', 'height_mean', 'pixcel_numbers_mean', 'rumor_scores_mean', 'pic_len']
indexs = []
samples = []
for idx in all_data.index:
    if str(all_data.iloc[idx].piclist) == 'nan':
        continue
    pic_list = all_data.loc[idx].piclist.split('\t')
    heights, widths, true_scores, rumor_scores = [], [], [], []
    pixcel_numbers = []
    for pic in pic_list:
        heights.append(mp_pid_height.get(pic, -1))
        widths.append(mp_pid_width.get(pic, -1))
        true_scores.append(mp_pid_TRUE.get(pic, -1))
        rumor_scores.append(mp_pid_rumor.get(pic, -1))
        pixcel_numbers.append(mp_pid_height.get(pic, -1) * mp_pid_width.get(pic, -1))
    indexs.append(idx)
    a_sample = {'height_mean': np.mean(heights), 
               'width_mean': np.mean(widths),
               'true_scores_mean': np.mean(true_scores),
               'rumor_scores_mean': np.mean(rumor_scores),
               'pixcel_numbers_mean': np.mean(pixcel_numbers),
               'pic_len' : len(pic_list)}
    samples.append(a_sample)
df_image_features = pd.DataFrame(samples, index=indexs)
df_image_features.head()

# all_data = all_data.fillna(-1)
province_list = ['北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林','黑龙江', '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东',\
                '河南', '湖北', '湖南', '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏',\
                '新疆', '台湾', '香港', '澳门']


province = all_data.userLocation.apply(get_province)

feature = ['userFollowCount', 'userFansCount', 'userWeiboCount', 
           'feature_len_piclist', 
           #'feature_userGender', 
           'feature_userLocation', 
           'feature_userProvince', 'feature_category', 'feature_follow_fan_rate']
text_features = ['feature_len_userDescription', 
                 'feature_userDescription_type', 
                 'feature_len_text', 'feature_pic_text_rate',
                ]
text_features.extend(tf_idf_news_columns_names)
text_features.extend(tf_idf_user_columns_names)
text_features.extend(lda_news_columns_names)
text_features.extend(lda_user_columns_names)
feature.extend(text_features)
feature.extend(features_news_location)
feature.extend(news_label_feature)
feature.extend(image_features)
# feature.extend(feature_bert)

#结构化特征
all_data['feature_len_piclist'] = all_data.piclist.apply(lambda x : (x if str(x) == 'nan' else len(x.split('\t'))))
# all_data['feature_userGender'] = all_data.userGender.apply(lambda x : {'男': 0, '女':1, -1:-1}[x])
all_data['feature_userLocation'] = pd.factorize(all_data.userLocation)[0]
all_data['feature_userProvince'] = pd.factorize(province)[0]
all_data['feature_category'] = pd.factorize(all_data.category)[0]
all_data['feature_follow_fan_rate'] = all_data.apply(lambda x : x.userFollowCount / (x.userFansCount + 0.001), axis = 1)

#文本特征
#人物描述
def get_user_type(description):
    """
    """
    official_names = ['官方', '法人', '办公室']
    self_media_names = ['自媒体', '头条文章作者', '媒体人']
    if str(description) == 'nan':
        return description
    if '医师' in description:
        return 1
    elif '专家' in description:
        return 2
    elif '演员' in description:
        return 3
    for name in official_names:
        if name in description:
            return 4
    for name in self_media_names:
        if name in description:
            return 5
    return 0
    
all_data['feature_len_userDescription'] = all_data.userDescription.apply(lambda x : (x if str(x) == 'nan' else len(x)))
all_data['feature_userDescription_type'] = all_data.userDescription.apply(get_user_type)

#新闻news
all_data['feature_len_text'] = all_data.text.apply(lambda x : (-1 if x == -1 else len(x)))
all_data['feature_pic_text_rate'] = all_data.apply(lambda x :  0 if x.feature_len_piclist == -1 else x.feature_len_piclist / (x.feature_len_text + 0.), axis = 1)

categorical_feature = ['feature_userGender', 'feature_userLocation', 'feature_userProvince', 'feature_category', 'feature_userDescription_type']
df_tf_idf_news_svd.index = all_data.index
df_tf_idf_user_svd.index = all_data.index
df_lda_user.index = all_data.index
df_lda_news.index = all_data.index
all_data_with_text_vector =  pd.concat([all_data, df_tf_idf_news_svd, df_tf_idf_user_svd, df_lda_user, df_lda_news, df_image_features], axis=1)
print(all_data_with_text_vector.head())

test = pd.read_csv('../data/task3/test_feature.csv')[feature]

lgb1_test = np.zeros(len(test))
for i in range(1, 21):
    model = lgb.Booster(model_file='../model/lgb_%d.model' % i)
    lgb1_test += model.predict(test)
lgb1_test /= 20.

lgb2_test = np.zeros(len(test))
for i in range(21, 41):
    model = lgb.Booster(model_file='../model/lgb_%d.model' % i)
    lgb2_test += model.predict(test)
lgb2_test /= 20.

lgb3_test = np.zeros(len(test))
for i in range(41, 61):
    model = lgb.Booster(model_file='../model/lgb_%d.model' % i)
    lgb3_test += model.predict(test)
lgb3_test /= 20.

lgb4_test = np.zeros(len(test))
for i in range(61, 81):
    model = lgb.Booster(model_file='../model/lgb_%d.model' % i)
    lgb4_test += model.predict(test)
lgb4_test /= 20.

lgb5_test = np.zeros(len(test))
for i in range(81, 101):
    model = lgb.Booster(model_file='../model/lgb_%d.model' % i)
    lgb5_test += model.predict(test)
lgb5_test /= 20.

model = lgb.Booster(model_file='../model/lgb_stack.model')

x_test_fusion = np.concatenate(( lgb1_test.reshape(-1, 1), lgb2_test.reshape(-1, 1), lgb3_test.reshape(-1, 1),
                         lgb4_test.reshape(-1, 1), lgb5_test.reshape(-1, 1)), axis=1)

fusion_predictions = model.predict(x_test_fusion, num_iteration=model.best_iteration) 
test_data['pre'] = fusion_predictions > 0.19
test_news_label = test.news_label.copy()
test_news_label.index = range(test_news_label.shape[0])
test_data['news_label'] = test_news_label
pre_res = list(test_data['pre']).copy() 
for i in range(test_data.shape[0]):
    count_value_test_pre = test_data[test_data.news_label == test_data.iloc[i].news_label].pre.value_counts()
    if len(count_value_test_pre) > 1:
        if (0. + count_value_test_pre[True]) / count_value_test_pre[False] >= 1.1:
            pre_res[i] = True
        elif (0. + count_value_test_pre[False]) / count_value_test_pre[True] >= 2:
            pre_res[i] = False
print (sum(test_data['pre'] != pre_res))
test_data['pre'] = pre_res

pre_by_model = []
true_label = []
def get_same_text_pre(test_sample):
    """
    """
    if test_sample['id'] in mp_test_id_pre:
        pre_by_model.append(test_sample['pre'])
        true_label.append(mp_test_id_pre[test_sample['id']])
        return mp_test_id_pre[test_sample['id']]
    return test_sample['pre']
print(test_data['pre'].value_counts())
print(test_data['pre'].value_counts()[0] / test_data['pre'].value_counts()[1], 0.994866476536168)
pre = test_data.apply(get_same_text_pre, axis=1)
print(pre.shape)
print(test_data.shape)
test_data['label'] = list(pre)
test_data['label'] = test_data['label'].astype('float')
test_data[['id', 'label']].to_csv('../submit/submit_test.csv', index=False)
print(precision_recall_fscore_support(pre_by_model, true_label))
print(f1_score(pre_by_model, true_label, average='macro'))
