# models/kpi2_ticket/train_tree.py
import os, sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ==== Rutas para los archivos ====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   
BASE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))   
DB_PATH  = os.path.join(BASE_DIR, "db", "talita.db")
OUT_DIR  = THIS_DIR
os.makedirs(OUT_DIR, exist_ok=True)
print("Usando BD:", DB_PATH)

# ==== 1) Dataset diario (fecha + modalidad) ====
conn = sqlite3.connect(DB_PATH)
query = """
WITH tt AS (
  SELECT
    DATE(fecha)          AS fecha,
    ticket_id,
    MAX(modalidad)       AS modalidad,
    MAX(dia_semana)      AS dia_semana,
    SUM(monto_linea)     AS total_ticket
  FROM ventas
  GROUP BY fecha, ticket_id
),
daily AS (
  SELECT
    fecha,
    modalidad,
    dia_semana,
    ROUND(AVG(total_ticket), 2) AS ticket_promedio,     -- KPI base
    COUNT(*)                    AS num_clientes,
    ROUND(SUM(total_ticket), 2) AS total_ventas_dia     -- total por fecha+modalidad
  FROM tt
  GROUP BY fecha, modalidad, dia_semana
)
SELECT * FROM daily
ORDER BY fecha, modalidad;
"""
df = pd.read_sql_query(query, conn)
conn.close()

df.to_csv(os.path.join(OUT_DIR, "dataset_ticket_diario.csv"), index=False)
print("Filas (fecha+modalidad):", len(df))

# --- Resumen global por fecha (local + pensión combinados) ---
df_global = (
    df.groupby(["fecha","dia_semana"], as_index=False)
      .agg(
          ticket_promedio=("ticket_promedio","mean"),  # promedio entre modalidades
          num_clientes=("num_clientes","sum"),
          total_ventas_dia=("total_ventas_dia","sum")
      )
      .sort_values("fecha")
)
df_global.to_csv(os.path.join(OUT_DIR, "dataset_ticket_diario_global.csv"), index=False)

# ==== 2) dia_semana y modalidad ====
cat_cols = ["dia_semana", "modalidad"]
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = enc.fit_transform(df[cat_cols])
X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(cat_cols))

X = pd.concat([df[["num_clientes"]].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
y = df["ticket_promedio"].values

# ==== 3) Modelo (árbol de decisión) ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

# ==== 4) Evaluación ====
y_pred = tree.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R²: {r2:.4f} | RMSE: {rmse:.2f} soles")

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Ticket real (S/)")
plt.ylabel("Ticket predicho (S/)")
plt.title("Árbol de Decisión - Ticket Promedio Diario")
plt.axline((0,0), slope=1, color="red", linestyle="--")
plt.tight_layout()
plt.show()

# ==== 5) Exportables técnicos ====
pred = pd.DataFrame({
    "fecha": df.loc[X_test.index, "fecha"].values,
    "modalidad": df.loc[X_test.index, "modalidad"].values,
    "ticket_real": y_test,
    "ticket_predicho": y_pred
})
pred["residual"] = pred["ticket_real"] - pred["ticket_predicho"]
pred.to_csv(os.path.join(OUT_DIR, "predicciones_ticket.csv"), index=False)

importancias = pd.DataFrame({
    "feature": X.columns,
    "importance": tree.feature_importances_
}).sort_values("importance", ascending=False)
importancias.to_csv(os.path.join(OUT_DIR, "importancias.csv"), index=False)

rules = export_text(tree, feature_names=list(X.columns))
with open(os.path.join(OUT_DIR, "reglas_arbol.txt"), "w") as f:
    f.write(rules)
print("\nReglas del árbol:\n", rules)

# ==== 6) KPIs explicados para negocio ====
# 6.0 Predicción para todo el dataset (para residual global)
y_hat_all = tree.predict(X)
df_kpi = df.copy()
df_kpi["ticket_predicho"] = y_hat_all
df_kpi["residual"] = df_kpi["ticket_promedio"] - df_kpi["ticket_predicho"]  # +: mejor de lo esperado, -: peor

# 6.1 Clasificación de ticket (alto/medio/bajo) por percentiles globales
p25 = df_kpi["ticket_promedio"].quantile(0.25)
p50 = df_kpi["ticket_promedio"].quantile(0.50)
p75 = df_kpi["ticket_promedio"].quantile(0.75)
def clas_ticket(v):
    if v >= p75:  return "alto"
    if v >= p50:  return "medio"
    return "bajo"
df_kpi["clase_ticket"] = df_kpi["ticket_promedio"].apply(clas_ticket)

# 6.2 Reglas simples de recomendación
cli_p60 = df_kpi["num_clientes"].quantile(0.60)
def recomendacion(row):
    if row["modalidad"] == "local" and row["num_clientes"] >= cli_p60 and row["ticket_promedio"] <= p50:
        return "Local: alta afluencia con ticket bajo → lanzar combo/upsell en mostrador."
    if row["modalidad"] == "pension" and row["residual"] < 0:
        return "Pensión: ticket por debajo de lo esperado → revisar menú del día/porciones."
    if row["clase_ticket"] == "alto":
        return "Mantener propuesta de valor actual; documentar qué funcionó."
    return "Monitoreo sin cambios; evaluar micro-promos según afluencia."
df_kpi["recomendacion"] = df_kpi.apply(recomendacion, axis=1)

# 6.3 Resumen final para dashboard
resumen_final = df_kpi[[
    "fecha","modalidad","dia_semana","num_clientes",
    "ticket_promedio","ticket_predicho","residual","clase_ticket","recomendacion"
]].sort_values(["fecha","modalidad"])
resumen_final.to_csv(os.path.join(OUT_DIR, "kpi2_resumen_final.csv"), index=False)

# 6.4 Mejores días por modalidad (promedio ticket)
top_dias = (
    df_kpi.groupby(["modalidad","dia_semana"])["ticket_promedio"]
      .mean().round(2).reset_index()
      .sort_values(["modalidad","ticket_promedio"], ascending=[True, False])
)
top_dias.to_csv(os.path.join(OUT_DIR, "kpi2_top_dias.csv"), index=False)

# 6.4-bis Mejores días a NIVEL GLOBAL (sin modalidad)
top_dias_global = (
    df_global.groupby("dia_semana")["ticket_promedio"]
    .mean().round(2).sort_values(ascending=False).reset_index()
)
top_dias_global.to_csv(os.path.join(OUT_DIR, "kpi2_top_dias_global.csv"), index=False)

# 6.4-ter Top FECHAS con mayor ticket (útil para calendario/promos)
top_fechas = df_global.sort_values("ticket_promedio", ascending=False)[
    ["fecha","dia_semana","ticket_promedio","num_clientes","total_ventas_dia"]
].head(20)
top_fechas.to_csv(os.path.join(OUT_DIR, "kpi2_top_fechas.csv"), index=False)

# 6.5 Estabilidad por modalidad y día
stab = (
    df_kpi.groupby(["modalidad","dia_semana"])["ticket_promedio"]
      .agg(std="std", mean="mean", count="count").reset_index()
)
stab["cv_%"] = (stab["std"] / stab["mean"] * 100).round(1)
stab = stab.sort_values(["modalidad","cv_%"])
stab.to_csv(os.path.join(OUT_DIR, "kpi2_estabilidad_modalidad.csv"), index=False)

print("\nCSV explicativos generados en:", OUT_DIR)
print(" - dataset_ticket_diario.csv (base por modalidad)")
print(" - dataset_ticket_diario_global.csv (base global por fecha)")
print(" - importancias.csv, reglas_arbol.txt, predicciones_ticket.csv")
print(" - kpi2_resumen_final.csv  ← PARA INFORME")
print(" - kpi2_top_dias.csv       ← Top por modalidad")
print(" - kpi2_top_dias_global.csv← Top global por día de semana")
print(" - kpi2_top_fechas.csv     ← Fechas con mayor ticket")
print(" - kpi2_estabilidad_modalidad.csv")