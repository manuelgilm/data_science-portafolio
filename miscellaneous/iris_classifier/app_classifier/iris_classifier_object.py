import sklearn
import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier

class Iris_Classifier:
    '''Create a object to implement the iris classifier.
    '''
    def __init__(self, model_path:str):
        self.model = self.get_model(model_path)
        self.iris_species = {
            0:'Setosa',
            1:'Versicolour',
            2:'Virginica'
        }
    
    def get_model(self, model_path:str) -> MLPClassifier:
        '''Open the pkl file which store the model.
        Arguments: 
            model_path: Path model with pkl extension
        
        Returns:
            model: Model object
        '''

        with open(model_path,"rb") as f:
            model = pickle.load(f)
        
        return model

    def make_prediction(self, features:dict)->str:
        '''Predicts the species.
        Argument:
            features: list
        
        return:
            Species: str
        '''
        features = np.array(list(features.values()))
        pred = self.model.predict(features.reshape(1,-1))[0]
        species_pred = self.iris_species[pred]
        return species_pred

        
        