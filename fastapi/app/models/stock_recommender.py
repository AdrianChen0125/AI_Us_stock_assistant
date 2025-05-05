import mlflow.pyfunc

# Load the latest version (or you can load a fixed model URI)
MODEL_URI = "models:/stock_rc/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)