# models/modelo_unico/build_dataset_unificado.py

import os
import sqlite3
import pandas as pd
import numpy as np

# ============================================================
# 1) RUTAS
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../models/modelo_unico
BASE_DIR = os.path.dirname(os.path.dirname(THIS_DIR))      # .../algoritmo
DB_PATH  = os.path.join(BASE_DIR, "db", "talita_realista.db")
OUT_DIR  = THIS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

print("Usando BD:", DB_PATH)

conn = sqlite3.connect(DB_PATH)

# ============================================================
# 2) DATASET DETALLADO POR LÍNEA (BASE PARA LOS 3 KPIs)
#    - Une tickets + clientes + detalle_ticket + platos
# ============================================================
query_lineas = """
SELECT
    t.id_ticket,
    t.fecha,
    substr(t.fecha, 1, 7)           AS mes,
    t.modalidad,                    -- local / pension
    t.total                         AS total_ticket,
    t.ticket_promedio               AS ticket_promedio_ticket,
    c.id_cliente,
    c.nombre                        AS nombre_cliente,
    c.tipo_cliente,
    c.edad,
    d.id_plato,
    p.nombre                        AS nombre_plato,
    p.categoria                     AS categoria_plato,
    p.precio                        AS precio_plato,
    p.costo                         AS costo_plato,
    d.cantidad,
    d.subtotal,
    d.extra
FROM tickets t
JOIN clientes       c ON c.id_cliente = t.id_cliente
JOIN detalle_ticket d ON d.id_ticket  = t.id_ticket
JOIN platos         p ON p.id_plato   = d.id_plato
ORDER BY t.fecha, t.id_ticket;
"""

df_lines = pd.read_sql_query(query_lineas, conn)
print("Filas de detalle (líneas de ticket):", len(df_lines))

# Margen por línea
df_lines["margen_linea"] = df_lines["subtotal"] - df_lines["costo_plato"] * df_lines["cantidad"]

# ============================================================
# 3) KPIs DIARIOS (KPI-1 y KPI-2) DESDE TABLA TICKETS + MARGEN
#    - num_clientes
#    - ventas_totales_dia
#    - ticket_promedio_dia
#    - margen_total_dia
# ============================================================
query_diario = """
SELECT
    fecha,
    modalidad,
    COUNT(DISTINCT id_ticket)           AS num_clientes,
    SUM(total)                          AS ventas_totales_dia,
    AVG(ticket_promedio)                AS ticket_promedio_dia
FROM tickets
GROUP BY fecha, modalidad
ORDER BY fecha, modalidad;
"""

daily = pd.read_sql_query(query_diario, conn)

# margen_total_dia desde las líneas
margen_diario = (
    df_lines
    .groupby(["fecha", "modalidad"], as_index=False)["margen_linea"]
    .sum()
    .rename(columns={"margen_linea": "margen_total_dia"})
)

daily = daily.merge(
    margen_diario,
    on=["fecha", "modalidad"],
    how="left"
)

# columna de mes (YYYY-MM)
daily["mes"] = daily["fecha"].astype(str).str.slice(0, 7)

# redondeos suaves
daily["ventas_totales_dia"]  = daily["ventas_totales_dia"].round(2)
daily["ticket_promedio_dia"] = daily["ticket_promedio_dia"].round(2)
daily["margen_total_dia"]    = daily["margen_total_dia"].round(2)

print("Filas diarias (fecha+modalidad):", len(daily))

# ============================================================
# 4) PLATO MÁS VENDIDO POR MES (BASE PARA KPI-3)
#    Usa tabla rotacion_insumos (mes, id_plato, cantidad_vendida_mes)
# ============================================================
# Cargar catálogo de platos para poder obtener el nombre
df_platos = pd.read_sql_query(
    "SELECT id_plato, nombre, categoria, precio, costo FROM platos",
    conn
)

query_rot = """
SELECT mes, id_plato, cantidad_vendida_mes
FROM rotacion_insumos
ORDER BY mes, cantidad_vendida_mes DESC;
"""

df_rot = pd.read_sql_query(query_rot, conn)

if df_rot.empty:
    print("⚠️ Tabla rotacion_insumos está vacía. No habrá información de plato_top_mes.")
    df_top = pd.DataFrame(columns=["mes", "plato_top_mes", "cant_top_mes"])
else:
    # Top 1 plato por mes (según cantidad vendida)
    df_top = (
        df_rot
        .sort_values(["mes", "cantidad_vendida_mes"], ascending=[True, False])
        .groupby("mes")
        .head(1)
        .reset_index(drop=True)
    )

    # unir con nombre de plato
    df_top = df_top.merge(
        df_platos[["id_plato", "nombre"]],
        on="id_plato",
        how="left"
    )

    df_top.rename(
        columns={
            "nombre": "plato_top_mes",
            "cantidad_vendida_mes": "cant_top_mes"
        },
        inplace=True
    )

    df_top = df_top[["mes", "plato_top_mes", "cant_top_mes"]]

conn.close()

print("Meses con plato_top_mes calculado:", len(df_top))

# ============================================================
# 5) UNIFICAR:
#    - Detalle de líneas (cliente, plato, costos)
#    - KPIs diarios
#    - Plato más vendido por mes
# ============================================================
# Añadimos 'mes' a df_lines (ya viene en el SELECT, pero aseguramos)
df_lines["mes"] = df_lines["fecha"].astype(str).str.slice(0, 7)

# Merge con KPIs diarios (se repiten por cada línea de ese día/modalidad)
df_full = df_lines.merge(
    daily[["fecha", "modalidad", "mes",
           "num_clientes", "ventas_totales_dia",
           "ticket_promedio_dia", "margen_total_dia"]],
    on=["fecha", "modalidad", "mes"],
    how="left"
)

# Merge con plato más vendido del mes
df_full = df_full.merge(
    df_top,
    on="mes",
    how="left"
)

# Orden de columnas más amigable
cols_orden = [
    # contexto temporal
    "fecha", "mes", "modalidad",

    # cliente
    "id_cliente", "nombre_cliente", "tipo_cliente", "edad",

    # ticket
    "id_ticket", "total_ticket", "ticket_promedio_ticket",

    # línea de pedido
    "id_plato", "nombre_plato", "categoria_plato",
    "precio_plato", "costo_plato",
    "cantidad", "subtotal", "margen_linea", "extra",

    # KPIs diarios
    "num_clientes", "ventas_totales_dia",
    "ticket_promedio_dia", "margen_total_dia",

    # info mensual para KPI-3
    "plato_top_mes", "cant_top_mes"
]

# Nos aseguramos de usar solo las columnas que realmente existen
cols_existentes = [c for c in cols_orden if c in df_full.columns]
df_full = df_full[cols_existentes].copy()

# ============================================================
# 6) GUARDAR CSV UNIFICADO
# ============================================================
OUT_PATH = os.path.join(OUT_DIR, "dataset_unificado.csv")
df_full.to_csv(OUT_PATH, index=False)

print("\nDataset unificado generado en:", OUT_PATH)
print("Columnas:")
print(df_full.columns.tolist())
print("\nPrimeras filas:")
print(df_full.head(5).to_string(index=False))