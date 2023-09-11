from typing import List, Optional

import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

from src.preprocess import TrafficPreprocess

traffic_preprocess = TrafficPreprocess()
traffic_preprocess.agent_model_load()
traffic_preprocess.load_models_etap_02()
traffic_preprocess.load_models_etap_03()
traffic_preprocess.load_models_etap_04()

app = FastAPI()

class Item(BaseModel):
    data: str

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
def predict(items: List[Item]):
    
    print('+++++++')
    #print(requests)
    
    #requests = jsonable_encoder(requests)
    #input_data = request.input_data
    
    list_clusters = []
    list_event = []
    for item in items:
        #input_data = request.name
    
        print('============= item =========')
        print(item.data)
        
        json_dict = json.loads(item.data)
        
        print('============= item convert to dict =========')
        #print(json_dict)

        result_dict = traffic_preprocess.preprocess_stage_one(json_dict)

        list_event.append(result_dict['EVENT_ID'])

        if result_dict['RIGHT_CLIENT_IP'] != result_dict['CLIENT_IP']:
            #trafic_cluster 0 - Неверный IP адрес
            list_clusters.append(0)
            continue

        result_dict = traffic_preprocess.preprocess_stage_two(result_dict)
        futures = traffic_preprocess.clusters_create_futures_array(result_dict)
        trafic_class = int(traffic_preprocess.trafic_model_predict(futures)[0])

        if trafic_class == 0:
            #trafic_cluster 1 - Обычный трафик
            list_clusters.append(1)
            continue

        trafic_cluster = 1
        trafic_cluster = int(traffic_preprocess.trafic_cluster_predict(futures)[0])+2
        list_clusters.append(trafic_cluster)
    
    assert len(list_clusters) == len(list_event)

    predictions = []
    for cluster, event in zip(list_clusters,list_event):
        dict_result = {"EVENT_ID": event, "LABEL_PRED": cluster}
        #json_str = json.dumps(dict_result, ensure_ascii=False)
        predictions.append(dict_result)
    
    # Возвращаем предсказание в ответе
    return predictions #PredictResponse(data=predictions)