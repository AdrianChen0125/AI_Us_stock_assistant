import mlflow.pyfunc
import pandas as pd
import joblib

class StockRecommenderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        distances, indices = self.model.named_steps["nn"].kneighbors(model_input)
        return pd.DataFrame(indices, columns=[f"neighbor_{i+1}" for i in range(indices.shape[1])])