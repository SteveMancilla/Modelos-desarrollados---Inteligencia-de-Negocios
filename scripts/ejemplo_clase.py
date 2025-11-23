
import pandas as pd

from sklearn.linear_model import LogisticRegression

data_d = {
    "asis":[10, 5, 70,90],
    "aprueba":[0,0,1,1]
}

data = pd.DataFrame(data_d)
print(data)


#Crear modelo de regresion logistica
modelo = LogisticRegression()
modelo.fit(data[["asis"]], data["aprueba"])
print("Modelo entrenado: ", modelo.intercept_) #Beta 0
print("Coeficiente: ", modelo.coef_) #Beta 1


# Hacer predicciones
por_predicciopn = int(input("Ingrese el porcentaje de asistencia para predecir aprobacion: "))
prediccion = modelo.predict([[por_predicciopn]])
print("Prediccion para", por_predicciopn, "%","de asistencia: ", prediccion)

probabilidad = modelo.predict_proba([[por_predicciopn]])
print("Probabilidad de no aprobar y aprobar: ", probabilidad)