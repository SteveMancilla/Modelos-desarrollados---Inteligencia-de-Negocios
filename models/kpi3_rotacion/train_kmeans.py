# models/kpi3_rotacion/train_kmeans.py
import os, sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==== Rutas ====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
DB_PATH  = os.path.join(BASE_DIR, "db", "talita.db")
OUT_DIR  = THIS_DIR
os.makedirs(OUT_DIR, exist_ok=True)
print("Usando BD:", DB_PATH)

# ==== 1) Construir dataset mensual de rotación ====
conn = sqlite3.connect(DB_PATH)

query = """
WITH ventas_platos AS (
  SELECT DATE(v.fecha) AS fecha,
         STRFTIME('%Y-%m', v.fecha) AS mes,
         p.id_plato AS plato_id,
         SUM(v.cantidad) AS platos_vendidos
  FROM ventas v
  JOIN platos p ON p.nombre = v.item
  GROUP BY mes, DATE(v.fecha), p.id_plato
),
consumo_diario AS (
  SELECT vp.fecha, vp.mes, rd.insumo_id,
         SUM(vp.platos_vendidos * rd.cantidad) AS consumo
  FROM ventas_platos vp
  JOIN recetas_detalle rd ON rd.plato_id = vp.plato_id
  GROUP BY vp.fecha, vp.mes, rd.insumo_id
),
consumo_mensual AS (
  SELECT mes, insumo_id, SUM(consumo) AS consumo_total
  FROM consumo_diario
  GROUP BY mes, insumo_id
),
inventario_prom_mensual AS (
  SELECT STRFTIME('%Y-%m', fecha) AS mes,
         i.id_insumo AS insumo_id,
         AVG((stock_inicio + stock_fin)/2.0) AS inventario_promedio
  FROM inventario_diario inv
  JOIN insumos i ON i.nombre = inv.insumo
  GROUP BY mes, i.id_insumo
)
SELECT
  i.id_insumo,
  i.nombre AS insumo,
  i.unidad,
  ROUND(i.costo_unitario,2) AS costo_unitario,
  c.mes,
  COALESCE(c.consumo_total,0) AS consumo_total,
  ROUND(COALESCE(inv.inventario_promedio,0),4) AS inventario_promedio,
  CASE
    WHEN COALESCE(inv.inventario_promedio,0) > 0
      THEN COALESCE(c.consumo_total,0)/inv.inventario_promedio
    ELSE NULL
  END AS rotacion_mensual
FROM insumos i
LEFT JOIN consumo_mensual c ON c.insumo_id = i.id_insumo
LEFT JOIN inventario_prom_mensual inv ON inv.insumo_id = i.id_insumo AND inv.mes = c.mes
ORDER BY i.nombre, c.mes;
"""
df_mensual = pd.read_sql_query(query, conn)
conn.close()

df_mensual.to_csv(os.path.join(OUT_DIR, "kpi3_rotacion_mensual.csv"), index=False)
print(f"Insumos-meses registrados: {len(df_mensual)}")

# ==== 2) Promedio mensual por insumo (para K-Means) ====
df_prom = (df_mensual.groupby(["insumo","unidad","costo_unitario"], as_index=False)
            .agg(rotacion_mensual_prom=("rotacion_mensual","mean"),
                 consumo_total_prom=("consumo_total","mean"),
                 inventario_promedio=("inventario_promedio","mean")))

# Quitamos nulos
df_model = df_prom.dropna(subset=["rotacion_mensual_prom"]).copy()

# ==== 3) Preparar features ====
features = df_model[["rotacion_mensual_prom","consumo_total_prom","costo_unitario"]].astype(float)
scaler   = StandardScaler()
X        = scaler.fit_transform(features)

# ==== 4) K-Means (k=3) ====
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
df_model["cluster"] = kmeans.fit_predict(X)

orden = (df_model.groupby("cluster")["rotacion_mensual_prom"]
          .mean().sort_values(ascending=False).index.tolist())
map_etiqueta = {orden[0]:"alta", orden[1]:"media", orden[2]:"baja"}
df_model["clase_rotacion"] = df_model["cluster"].map(map_etiqueta)

# ==== 5) Exportables ====
cols_out = ["insumo","unidad","costo_unitario","consumo_total_prom",
            "inventario_promedio","rotacion_mensual_prom","clase_rotacion"]

df_model[cols_out].sort_values(["clase_rotacion","rotacion_mensual_prom"],
                               ascending=[True,False]) \
       .to_csv(os.path.join(OUT_DIR, "kpi3_clusters_insumos_mensual.csv"), index=False)

# Centroides interpretables
centroides = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=["rotacion_mensual_prom","consumo_total_prom","costo_unitario"]
)
centroides["cluster"] = range(3)
centroides["clase_rotacion"] = centroides["cluster"].map(map_etiqueta)
centroides.to_csv(os.path.join(OUT_DIR, "kpi3_centroides_mensual.csv"), index=False)

print("\nCSV generados en:", OUT_DIR)
print(" - kpi3_rotacion_mensual.csv (rotación mensual por insumo)")
print(" - kpi3_clusters_insumos_mensual.csv (promedio mensual + clase)")
print(" - kpi3_centroides_mensual.csv (centroides)")