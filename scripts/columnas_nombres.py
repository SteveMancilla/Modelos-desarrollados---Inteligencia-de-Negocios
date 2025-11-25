import pandas as pd

pred = pd.read_csv("models/modelo_unico/predicciones_modelo_unico.csv")
print(pred.columns.tolist())