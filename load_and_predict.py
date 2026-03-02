import mlflow.sklearn
from sklearn.datasets import load_wine

model_name = "WineRF"
model_version = 2 
model_uri = "file:///D:/7 sem/mashine learn/mlruns/4/models/m-91a7fc1db45e45058f95ddcf7ee23549/artifacts"


loaded_model = mlflow.sklearn.load_model(model_uri)


X, y = load_wine(return_X_y=True)
sample_data = X[:1000] 


predictions = loaded_model.predict(sample_data)
print("Predictions:", predictions)