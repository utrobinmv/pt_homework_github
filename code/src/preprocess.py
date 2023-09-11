from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pickle
import re
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction import text
from sklearn import cluster

#Для определения стран IP адресов
from pysxgeo import sxgeo

import torch
from transformers import AutoModel, AutoTokenizer
import catboost as cb

from src.constants import CONST_PATH_SxGeo, CONST_USER_AGENT_N_CLUSTERS
from src.constants import CONST_PATH_MODELS
from src.constants import CONST_TRAFIC_N_CLUSTER

class TrafficPreprocess:
    def __init__(self) -> None:
        '''
        Класс используется для предобработки данных трафика
        '''
        self.geo = sxgeo.SxGeo(CONST_PATH_SxGeo)
        self.valid_MATCHED_VARIABLE_SRC = ['REQUEST_GET_ARGS',
                                           'REQUEST_COOKIES',
                                           'REQUEST_HEADERS',
                                           'REQUEST_PATH',
                                           'REQUEST_ARGS',
                                           'RESPONSE_HEADERS',
                                           'REQUEST_POST_ARGS',
                                           'REQUEST_URI',
                                           'REQUEST_XML',
                                           'REQUEST_ARGS_KEYS',
                                           'REQUEST_JSON',
                                           'CLIENT_USERAGENT',
                                           'CLIENT_SESSION_ID',
                                           'REQUEST_QUERY',
                                           'RESPONSE_BODY',
                                           'REQUEST_CONTENT_TYPE',
                                           'REQUEST_FILES',
                                           'CLIENT_IP']
        
        self.valid_RESPONSE_CODE = ['200','404','302','403',
                                    '304','504','301','502',
                                    '500','400','204','206',
                                    '307','405','503','401',
                                    '303','207','429']
        
        #CLIENT_USERAGENT
        str_regex_agent_tokenizer = r'\b[а-яА-Яa-zA-Z]+\b'
        #self.agent_vectorizer = text.TfidfVectorizer(tokenizer=lambda text: re.findall(str_regex_agent_tokenizer, text),min_df=2)
        self.agent_vectorizer = text.TfidfVectorizer(token_pattern=str_regex_agent_tokenizer,min_df=2)
        self.agent_kmeans = cluster.KMeans(n_clusters=CONST_USER_AGENT_N_CLUSTERS, random_state=0)
        
        
    def load_models_etap_02(self) -> None:
        '''
        Загружает модели для кластеризации агентов и получения эмбедингов
        '''
        #embedding for MATCHED_VARIABLE_NAME and MATCHED_VARIABLE_VALUE
        model_name = 'cointegrated/rubert-tiny2'
        self.traffic_tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=False)
        self.traffic_model = AutoModel.from_pretrained(model_name,local_files_only=False)

        #Кластеризация аномального трафика
        self.trafic_kmeans = cluster.KMeans(n_clusters=CONST_TRAFIC_N_CLUSTER, random_state=0)

    def load_models_etap_03(self) -> None:
        '''
        Загружает CatBoost модель выявления аномалий в трафике
        '''
        #Кластеризация
        self.trafic_model = cb.CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, loss_function='Logloss',eval_metric='Accuracy')
        self.trafic_model.load_model("models/cb_model_traffic.cb")

    def load_models_etap_04(self) -> None:
        '''
        Загружает Модель кластеризации аномального трафика
        '''
        self.trafic_kmeans = pickle.load(open(CONST_PATH_MODELS + 'trafic_kmeans_model.pkl', 'rb'))

    def check_right_ip(self,ip: Optional[str]) -> Optional[str]:
        '''
        Проверка корректность IPv4 и IPv6
        '''
        #regex на поиск IPv4 и IPv6
        pattern = r'(\b(?:(?:\d{1,3}\.){3}\d{1,3}|(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}|(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4})\b)'

        ip_address = re.search(pattern, str(ip))
    
        if ip_address:
            return ip_address.group(0)
        else:
            return None
        
    def get_country(self, ip: Optional[str]) -> Optional[str]:
        '''
        Получает страну из ip
        '''
        result = None
        try:
            result = self.geo.get_country(str(ip))
        except:
            pass
        return result  
    
    def check_valid_matched_variable_src(self,x: Optional[str]) -> bool:
        '''
        Проверка MATCHED_VARIABLE_SRC на валидноcть значения
        '''
        return x in self.valid_MATCHED_VARIABLE_SRC

    def check_valid_response_code(self,x: Optional[str]) -> bool:
        '''
        Проверка RESPONSE_CODE на валидноcть значения
        '''
        return x in self.valid_RESPONSE_CODE

    def agent_str_preproc(self, x: Optional[str]) -> bool:
        '''
        Препроцессинг данных агента, перед подачей в корпус
        '''
        return str(x).lower()

    def agent_model_fit(self,preproc_corpus: np.ndarray) -> None:
        '''
        Обучаем модель 
        '''
        vectors = self.agent_vectorizer.fit_transform(preproc_corpus)
        self.agent_kmeans.fit(vectors)
        
    def agent_model_save(self) -> None:
        '''
        Сохранение модели кластеризации USER AGENT
        '''
        pickle.dump(self.agent_vectorizer, open(CONST_PATH_MODELS + 'agent_tfidf_model.pkl', 'wb'))
        pickle.dump(self.agent_kmeans, open(CONST_PATH_MODELS + 'agent_kmeans_model.pkl', 'wb'))

    def agent_model_load(self) -> None:
        '''
        Загрузка модели кластеризации USER AGENT
        '''
        self.agent_vectorizer = pickle.load(open(CONST_PATH_MODELS + 'agent_tfidf_model.pkl', 'rb'))
        self.agent_kmeans = pickle.load(open(CONST_PATH_MODELS + 'agent_kmeans_model.pkl', 'rb'))


    def agent_model_predict(self, preproc_x: str) -> Dict[str, Union[csr_matrix, int]]:
        '''
        Предсказываем кластер user агента
        '''
        vector = self.agent_vectorizer.transform([preproc_x])
        cluster = self.agent_kmeans.predict(vector)
        dict_result = {'agent_vector': vector[0], 'agent_cluster': cluster[0]}
        return dict_result
        
    def request_size_preproc(self, x: Optional[str]) -> int:
        '''
        Препроцессинг данных request_size
        '''
        return int(x)

    def response_code_preproc(self, x: Optional[str]) -> int:
        '''
        Препроцессинг данных RESPONSE_CODE
        '''
        return int(x)
        
    def embed_cls(self, text: Optional[str]) -> np.ndarray:
        if isinstance(text, str):
            pass
        elif isinstance(text, float):        
            #print('from float:', text)
            #text = str(text)
            text = ''
        elif not isinstance(text, str):
            print(type(text))
        
        t = self.traffic_tokenizer(text, padding=True, truncation=True, return_tensors='pt') 
        t = {k: v.to(self.traffic_model.device) for k, v in t.items()}
        with torch.no_grad():
            model_output = self.traffic_model(**t)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()
    
    def preprocess_stage_one(self, request_dict: Dict[str, Any]):
        '''
        Препроцессинг входных данных первая стадия
        '''
        if isinstance(request_dict, dict):

            request_dict['RIGHT_CLIENT_IP'] = self.check_right_ip(request_dict['CLIENT_IP'])

            if request_dict['RIGHT_CLIENT_IP'] == request_dict['CLIENT_IP']:
                request_dict['preproc_CLIENT_IP_country'] = self.get_country(request_dict['CLIENT_IP'])
                request_dict['preproc_CLIENT_USERAGENT'] = self.agent_str_preproc(request_dict['CLIENT_USERAGENT'])
                request_dict['preproc_REQUEST_SIZE'] = self.request_size_preproc(request_dict['REQUEST_SIZE'])
                request_dict['preproc_RESPONSE_CODE_valid'] = self.check_valid_response_code(request_dict['RESPONSE_CODE'])
                request_dict['preproc_RESPONSE_CODE'] = self.response_code_preproc(request_dict['RESPONSE_CODE'])
                request_dict['preproc_MATCHED_VARIABLE_SRC_valid'] = self.check_valid_matched_variable_src(request_dict['MATCHED_VARIABLE_SRC'])

            pass
        else:
            print('Preprocess_stage_one error type:',type(text))
        
        return request_dict

    def preprocess_stage_two(self, request_dict: Dict[str, Any]):
        '''
        Препроцессинг входных данных вторая стадия
        '''
        if isinstance(request_dict, dict):
            if request_dict['RIGHT_CLIENT_IP'] == request_dict['CLIENT_IP']:
                request_dict.update(self.agent_model_predict(request_dict['preproc_CLIENT_USERAGENT']))

                request_dict['matched_var_name'] = self.embed_cls(request_dict['MATCHED_VARIABLE_NAME'])
                request_dict['matched_var_value'] = self.embed_cls(request_dict['MATCHED_VARIABLE_VALUE'])

        return request_dict


    def clusters_create_futures_array(self, request_dict: Dict[str, Any]) -> np.ndarray:
        '''
        Подготовить таблицу признаков
        '''
        preproc_RESPONSE_CODE_valid = np.array(int(request_dict['preproc_RESPONSE_CODE_valid']))
        preproc_MATCHED_VARIABLE_SRC_valid = np.array(int(request_dict['preproc_MATCHED_VARIABLE_SRC_valid']))

        futures = np.hstack((
        np.array(request_dict['preproc_RESPONSE_CODE']),
        np.array(request_dict['preproc_REQUEST_SIZE']),
        preproc_RESPONSE_CODE_valid,preproc_MATCHED_VARIABLE_SRC_valid,
        request_dict['matched_var_name'],
        np.array(request_dict['agent_cluster']),
        request_dict['matched_var_value'],
        ))
         
        return futures

    def trafic_model_predict(self, record: np.array):
        '''
        Классификация трафика на аномальный и обычный
        1 - аномальный, 0 - обычный
        '''
        result = self.trafic_model.predict([record])
        return result

    def trafic_cluster_predict(self, record: np.array):
        '''
        Кластеризация аномального трафика по кластерам
        '''
        result = self.trafic_kmeans.predict([record])

        return result       