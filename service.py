import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
iris_model_runner= bentoml.sklearn.get("iris_model:latest").to_runner()
svc=bentoml.Service("iris_classifier",runners=[iris_model_runner])
@svc.api(input=NumpyNdarray(),output=NumpyNdarray())

def classify(input_series:np.ndarray) -> np.ndarray:
    result=iris_model_runner.predict.run(input_series)
    return result
    
