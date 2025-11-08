# models/kpi1_margen/train_regresion.py
import os, sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ----------------------------
# Importacion de datos de la base de datos
# ----------------------------
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))           
BASE_DIR  = os.path.dirname(os.path.dirname(THIS_DIR))             
DB_PATH   = os.path.join(BASE_DIR, "db", "talita.db")
OUT_DIR   = THIS_DIR    # ubicacion para guardar CSV
os.makedirs(OUT_DIR, exist_ok=True)
print("Usando BD:", DB_PATH)

# ----------------------------
# 1) Cargar dataset desde SQLite
#    (agregamos ventas y costo totales para KPI)
# ----------------------------
conn = sqlite3.connect(DB_PATH)
query = """
SELECT 
    v.fecha,
    v.dia_semana,
    v.item AS tipo_plato,
    SUM(v.costo_linea)                    AS costo_insumos,
    SUM(v.cantidad)                       AS cantidad_vendida,
    SUM(v.monto_linea)                    AS ventas_totales,
    SUM(v.monto_linea - v.costo_linea)    AS margen_ganancia
FROM ventas v
JOIN platos p ON p.nombre = v.item
GROUP BY v.fecha, v.item, v.dia_semana
ORDER BY v.fecha;
"""
df = pd.read_sql_query(query, conn)
conn.close()

print("Registros cargados:", len(df))
print(df.head())

# ----------------------------
# 2) Preprocesamiento
# ----------------------------

categorical_cols = ["dia_semana", "tipo_plato"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Dataset final para el modelo
df_model = pd.concat(
    [df[["costo_insumos", "cantidad_vendida", "margen_ganancia"]], encoded_df],
    axis=1
)

X = df_model.drop(columns=["margen_ganancia"])
y = df_model["margen_ganancia"]

# ----------------------------
# 3) Split y entrenamiento
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 4) Evaluación
# ----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nR²: {r2:.4f}")
print(f"RMSE: {rmse:.2f} soles")

# Gráfico de ajuste
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Margen real (S/)")
plt.ylabel("Margen predicho (S/)")
plt.title("Ajuste del modelo de Regresión Lineal")
plt.axline((0, 0), slope=1, color="red", linestyle="--")
plt.tight_layout()
plt.show()

# ----------------------------
# 5) Guardar resultados del modelo
# ----------------------------
results = pd.DataFrame({
    "Margen_real": y_test.values,
    "Margen_predicho": y_pred
})
results.to_csv(os.path.join(OUT_DIR, "resultados.csv"), index=False)
print("Resultados guardados en:", os.path.join(OUT_DIR, "resultados.csv"))

coefs = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_
}).sort_values(by="Coeficiente", ascending=False)
coefs.to_csv(os.path.join(OUT_DIR, "coeficientes.csv"), index=False)
print("Coeficientes guardados en:", os.path.join(OUT_DIR, "coeficientes.csv"))

# ----------------------------
# 6) Análisis accionable por plato
#    - Residuales (real - predicho) para detectar sub/sobre desempeño
#    - Ranking de rentabilidad (margen unitario)
#    - Candidatos a ajuste de precio (alta venta + residual negativo)
# ----------------------------

# Predicción (y residual) para todo el dataset (no sólo test)
y_pred_all = model.predict(X)
df_pred = df.copy()
df_pred["margen_predicho"] = y_pred_all
df_pred["residual"] = df_pred["margen_ganancia"] - df_pred["margen_predicho"]  # >0 rinde más que lo esperado; <0 rinde menos

# Métricas por plato
agg = df_pred.groupby("tipo_plato").agg(
    ventas_totales=("ventas_totales","sum"),
    costo_total=("costo_insumos","sum"),
    margen_total=("margen_ganancia","sum"),
    cantidad_total=("cantidad_vendida","sum"),
    residual_prom=("residual","mean"),
    residual_total=("residual","sum"),
    n_obs=("residual","count"),
).reset_index()

# Derivadas
agg["precio_prom_por_ud"] = (agg["ventas_totales"] / agg["cantidad_total"]).replace([np.inf, -np.inf], np.nan)
agg["costo_prom_por_ud"]  = (agg["costo_total"]  / agg["cantidad_total"]).replace([np.inf, -np.inf], np.nan)
agg["margen_unitario"]    = (agg["margen_total"] / agg["cantidad_total"]).replace([np.inf, -np.inf], np.nan)

# Percentiles para cortes (robustos)
ventas_mediana = agg["ventas_totales"].median()
ventas_p60     = agg["ventas_totales"].quantile(0.60)
margen_p25     = agg["margen_unitario"].quantile(0.25)

# 6.1) Top rentables: alto margen unitario + ventas >= mediana
top_rentables = (
    agg[agg["ventas_totales"] >= ventas_mediana]
    .sort_values(["margen_unitario","ventas_totales"], ascending=False)
    .head(10)
    [["tipo_plato","ventas_totales","cantidad_total","margen_total","margen_unitario","precio_prom_por_ud","costo_prom_por_ud","residual_prom","n_obs"]]
)

# 6.2) Menos rentables: bajo margen unitario + ventas >= mediana
menos_rentables = (
    agg[agg["ventas_totales"] >= ventas_mediana]
    .sort_values(["margen_unitario","ventas_totales"], ascending=[True, False])
    .head(10)
    [["tipo_plato","ventas_totales","cantidad_total","margen_total","margen_unitario","precio_prom_por_ud","costo_prom_por_ud","residual_prom","n_obs"]]
)

# 6.3) Ajuste de precio: alta venta (>= p60) + residual NEGATIVO (rinde menos de lo esperado)
#      Sugerimos subir precio para cerrar el gap del residual por unidad.
cand_ajuste = agg[
    (agg["ventas_totales"] >= ventas_p60) & (agg["residual_prom"] < 0)
].copy()

# Propuesta de nuevo precio:
# - gap por unidad = (- residual_total / cantidad_total)
# - precio_sugerido = precio_prom_por_ud + max(0, gap_ud)
# - garantizamos precio >= costo + 1 sol
gap_ud = (-cand_ajuste["residual_total"] / cand_ajuste["cantidad_total"]).clip(lower=0).fillna(0)
precio_sugerido = cand_ajuste["precio_prom_por_ud"] + gap_ud
precio_minimo = cand_ajuste["costo_prom_por_ud"] + 1.0
cand_ajuste["precio_sugerido"] = np.maximum(precio_sugerido, precio_minimo).round(2)

ajuste_precio = cand_ajuste.sort_values(["ventas_totales","residual_prom"], ascending=[False, True]).head(10)[
    ["tipo_plato","ventas_totales","cantidad_total","precio_prom_por_ud","costo_prom_por_ud","margen_unitario","residual_prom","precio_sugerido","n_obs"]
]

# ----------------------------
# 7) Exportar tablas resumen
# ----------------------------
top_rentables.to_csv(os.path.join(OUT_DIR, "top_rentables.csv"), index=False)
menos_rentables.to_csv(os.path.join(OUT_DIR, "menos_rentables.csv"), index=False)
ajuste_precio.to_csv(os.path.join(OUT_DIR, "candidatos_ajuste_precio.csv"), index=False)

print("\nTablas generadas:")
print(" -", os.path.join(OUT_DIR, "top_rentables.csv"))
print(" -", os.path.join(OUT_DIR, "menos_rentables.csv"))
print(" -", os.path.join(OUT_DIR, "candidatos_ajuste_precio.csv"))

# (Opcional) Vista rápida en consola
print("\n=== TOP RENTABLES (margen alto y buena venta) ===")
print(top_rentables.head(10).to_string(index=False))

print("\n=== MENOS RENTABLES (margen bajo y buena venta) ===")
print(menos_rentables.head(10).to_string(index=False))

print("\n=== CANDIDATOS A AJUSTE DE PRECIO (alta venta + residual negativo) ===")
print(ajuste_precio.head(10).to_string(index=False))