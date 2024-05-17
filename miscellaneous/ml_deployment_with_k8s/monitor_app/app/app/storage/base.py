from typing import Dict 
from typing import Any
from typing import Union
from datetime import datetime
import pickle
from app.utils.utils import get_root_dir
import pandas as pd
from pathlib import Path 
import json 
import time 
DATA_STORAGE = get_root_dir() / "app/data"


class Prediction:

    def __init__(self, model_name:str):
        self.model_name = model_name
        self.filepath = DATA_STORAGE / self.model_name

    def save_value(self,value:Union[int, float]):

        data = self.__load_json()
        if not data:
            data = self.__create_json()
        print(data)
        new_data = self.__update_data(data, value)
        self.__save_json(new_data)

    def load_json(self):
        return self.__load_json()

    def __load_json(self):
        
        if self.filepath.with_suffix(".json").is_file():
            with open(self.filepath.with_suffix(".json"), "r") as f:
                data = json.load(f)
            return data 
        
    def __create_json(self):
        
        data = {
            "timestamp":[],
            "prediction":[]
        }
        with open(self.filepath.with_suffix(".json"), "w") as f:
            json.dump(data, f)

        return data

    def __update_data(self, data, value)->Dict[str, Any]:
        data.update({
            "timestamp":data["timestamp"] + [str(datetime.now())],
            "prediction":data["prediction"] + [value]
        })
        return data
    
    def __save_json(self, json_data):
         
        with open(self.filepath.with_suffix(".json"), "w") as f:
            json.dump(json_data, f)      


    def save(self)->None:
        with open(self.filepath.withsuffix(".pkl"), "wb") as f:
            pickle.dump(obj=self, file=f)
