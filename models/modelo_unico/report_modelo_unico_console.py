# models/modelo_unico/report_modelo_unico_console.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas a los CSV por KPI
KPI1_PATH = os.path.join(THIS_DIR, "resultados_kpi1_margen_plato.csv")
KPI2_PATH = os.path.join(THIS_DIR, "resultados_kpi2_ticket_diario.csv")
KPI3_PATH = os.path.join(THIS_DIR, "resultados_kpi3_rotacion.csv")
PRED_ALL_PATH = os.path.join(THIS_DIR, "predicciones_modelo_unico.csv")

# ===========================================================
# Carga de datos con validaciones básicas
# ===========================================================
def safe_read_csv(path, label):
    if not os.path.exists(path):
        print(f"[AVISO] No se encontró {label}: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Filas en {label}: {len(df)}")
    print(f"Columnas: {list(df.columns)}\n")
    return df

df_kpi1 = safe_read_csv(KPI1_PATH, "resultados_kpi1_margen_plato.csv")
df_kpi2 = safe_read_csv(KPI2_PATH, "resultados_kpi2_ticket_diario.csv")
df_kpi3 = safe_read_csv(KPI3_PATH, "resultados_kpi3_rotacion.csv")

print("\n" + "=" * 80)
print("REPORTE INTEGRADO – MODELO ÚNICO (Árbol de Decisión Multi-Output)")
print("=" * 80)

# ======================================================================================
# KPI 1: MARGEN DE GANANCIA POR PLATO
# ======================================================================================
if df_kpi1 is not None:
    print("\n" + "=" * 80)
    print("KPI-1: MARGEN DE GANANCIA POR PLATO")
    print("=" * 80)

    # Esperamos columnas: margen_plato_real, margen_plato_pred
    if not {"margen_plato_real", "margen_plato_pred"}.issubset(df_kpi1.columns):
        print("[KPI1] No se encontraron columnas 'margen_plato_real' y 'margen_plato_pred'.")
    else:
        # Residual por línea
        df_kpi1["res_margen_plato"] = df_kpi1["margen_plato_real"] - df_kpi1["margen_plato_pred"]

        # --- Resumen estadístico global ---
        stats_margen = df_kpi1[["margen_plato_real", "margen_plato_pred", "res_margen_plato"]].describe()
        print("\nResumen estadístico global del margen por plato (real vs predicho):")
        print(stats_margen)

        # --- Top 10 líneas con margen real > esperado ---
        cols_linea = ["fecha", "modalidad", "nombre_plato",
                      "margen_plato_real", "margen_plato_pred", "res_margen_plato"]
        cols_linea = [c for c in cols_linea if c in df_kpi1.columns]

        top_plus = df_kpi1.sort_values("res_margen_plato", ascending=False).head(10)[cols_linea]
        print("\nLíneas donde el margen REAL es mayor al esperado (TOP 10):")
        print(top_plus.to_string(index=False))

        # --- Top 10 líneas con margen real < esperado ---
        top_minus = df_kpi1.sort_values("res_margen_plato").head(10)[cols_linea]
        print("\nLíneas donde el margen REAL es menor al esperado (TOP 10):")
        print(top_minus.to_string(index=False))

        # --- Promedio de margen por plato ---
        if "nombre_plato" in df_kpi1.columns:
            margen_plato_agg = (
                df_kpi1
                .groupby("nombre_plato")[["margen_plato_real", "margen_plato_pred"]]
                .mean()
                .sort_values("margen_plato_real", ascending=False)
                .round(2)
                .head(15)
            )
            print("\nTop 15 platos con mayor margen promedio REAL:")
            print(margen_plato_agg.to_string())

        # ---------- Gráfico: real vs predicho (margen por plato) ----------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x=df_kpi1["margen_plato_real"],
            y=df_kpi1["margen_plato_pred"]
        )
        mmin = df_kpi1["margen_plato_real"].min()
        mmax = df_kpi1["margen_plato_real"].max()
        plt.plot([mmin, mmax], [mmin, mmax], linestyle="--")
        plt.title("Margen por plato: real vs predicho")
        plt.xlabel("Margen real por plato (S/)")
        plt.ylabel("Margen predicho por plato (S/)")
        plt.tight_layout()
        plt.show()

        # ---------- Histograma margen real ----------
        plt.figure(figsize=(7, 5))
        sns.histplot(df_kpi1["margen_plato_real"], kde=True)
        plt.title("Distribución del margen de ganancia por plato (real)")
        plt.xlabel("Margen por plato (S/)")
        plt.tight_layout()
        plt.show()

        # ---------- Barras: top 10 platos por margen promedio ----------
        if "nombre_plato" in df_kpi1.columns:
            top_plot = margen_plato_agg.head(10).reset_index()
            plt.figure(figsize=(8, 4))
            sns.barplot(data=top_plot, x="nombre_plato", y="margen_plato_real")
            plt.xticks(rotation=45, ha="right")
            plt.title("Top 10 platos por margen promedio REAL")
            plt.xlabel("Plato")
            plt.ylabel("Margen promedio (S/)")
            plt.tight_layout()
            plt.show()

# ======================================================================================
# KPI 2: TICKET PROMEDIO DIARIO
# ======================================================================================
if df_kpi2 is not None:
    print("\n" + "=" * 80)
    print("KPI-2: TICKET PROMEDIO DIARIO")
    print("=" * 80)

    # Buscamos columnas reales y predichas
    real_candidates = ["ticket_dia_real", "ticket_real", "ticket_promedio_dia"]
    pred_candidates = ["ticket_dia_pred", "ticket_pred", "ticket_predicho"]

    col_ticket_real = next((c for c in real_candidates if c in df_kpi2.columns), None)
    col_ticket_pred = next((c for c in pred_candidates if c in df_kpi2.columns), None)

    print(f"Columnas detectadas para KPI2 -> real: {col_ticket_real}, pred: {col_ticket_pred}")

    if col_ticket_real is None or col_ticket_pred is None:
        print("[KPI2] No se encontraron columnas adecuadas para ticket promedio real/predicho.")
    else:
        df_kpi2["res_ticket"] = df_kpi2[col_ticket_real] - df_kpi2[col_ticket_pred]

        # --- Resumen estadístico ---
        stats_ticket = df_kpi2[[col_ticket_real, col_ticket_pred, "res_ticket"]].describe()
        print("\nResumen estadístico del ticket promedio diario (real vs predicho):")
        print(stats_ticket)

        cols_ticket = ["fecha", "modalidad", col_ticket_real, col_ticket_pred, "res_ticket"]
        cols_ticket = [c for c in cols_ticket if c in df_kpi2.columns]

        # Días con ticket por debajo de lo esperado
        dias_bajos = (
            df_kpi2[df_kpi2["res_ticket"] < 0]
            .sort_values("res_ticket")
            .head(10)[cols_ticket]
        )
        print("\nDías con TICKET REAL por debajo del esperado (TOP 10):")
        print(dias_bajos.to_string(index=False))

        # Días con ticket por encima de lo esperado
        dias_altos = (
            df_kpi2[df_kpi2["res_ticket"] > 0]
            .sort_values("res_ticket", ascending=False)
            .head(10)[cols_ticket]
        )
        print("\nDías con TICKET REAL por encima del esperado (TOP 10):")
        print(dias_altos.to_string(index=False))

        # ---------- Gráfico: real vs predicho ----------
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            x=df_kpi2[col_ticket_real],
            y=df_kpi2[col_ticket_pred]
        )
        tmin = df_kpi2[col_ticket_real].min()
        tmax = df_kpi2[col_ticket_real].max()
        plt.plot([tmin, tmax], [tmin, tmax], linestyle="--")
        plt.title("Ticket promedio diario: real vs predicho")
        plt.xlabel("Ticket real (S/)")
        plt.ylabel("Ticket predicho (S/)")
        plt.tight_layout()
        plt.show()

        # ---------- Histograma ticket real ----------
        plt.figure(figsize=(7, 5))
        sns.histplot(df_kpi2[col_ticket_real], kde=True)
        plt.title("Distribución del ticket promedio diario (real)")
        plt.xlabel("Ticket promedio (S/)")
        plt.tight_layout()
        plt.show()

# ======================================================================================
# KPI 3: ROTACIÓN DE INVENTARIO (PLATOS TOP)
# ======================================================================================
if df_kpi3 is not None:
    print("\n" + "=" * 80)
    print("KPI-3: ROTACIÓN DE INVENTARIO (PLATOS TOP POR MES)")
    print("=" * 80)

    # Esperamos columnas: mes, plato_top_mes, cant_top_mes, rotacion_real, rotacion_pred
    needed = {"mes", "plato_top_mes", "cant_top_mes", "rotacion_real", "rotacion_pred"}
    if not needed.issubset(df_kpi3.columns):
        print(f"[KPI3] Faltan columnas en resultados_kpi3_rotacion.csv. Se esperaba: {needed}")
    else:
        # Mapeo 0/1/2 -> etiquetas
        label_map = {0: "baja", 1: "media", 2: "alta"}

        df_kpi3["rot_real_label"] = df_kpi3["rotacion_real"].map(label_map)
        df_kpi3["rot_pred_label"] = df_kpi3["rotacion_pred"].map(label_map)
        df_kpi3["acierto_rot"] = (df_kpi3["rotacion_real"] == df_kpi3["rotacion_pred"]).astype(int)

        # Distribución de clases reales
        dist_real = df_kpi3["rot_real_label"].value_counts().reindex(["alta", "media", "baja"]).fillna(0).astype(int)
        print("\nDistribución de clases de rotación (REAL):")
        print(dist_real)

        # Distribución de clases predichas
        dist_pred = df_kpi3["rot_pred_label"].value_counts().reindex(["alta", "media", "baja"]).fillna(0).astype(int)
        print("\nDistribución de clases de rotación (PREDICHA):")
        print(dist_pred)

        # Exactitud del modelo
        acc_rot = df_kpi3["acierto_rot"].mean()
        print(f"\nExactitud del modelo de rotación (platos top por mes): {acc_rot*100:.2f}%")

        # Top platos por cantidad vendida en el mes
        top_rot = (
            df_kpi3.sort_values("cant_top_mes", ascending=False)
            .head(15)[["mes", "plato_top_mes", "cant_top_mes", "rot_real_label", "rot_pred_label"]]
        )
        print("\nTop 15 platos más vendidos por mes (con clase de rotación):")
        print(top_rot.to_string(index=False))

        # ---------- Barras: distribución REAL ----------
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=dist_real.index,
            y=dist_real.values
        )
        plt.title("Distribución real de clases de rotación (platos top)")
        plt.xlabel("Clase de rotación")
        plt.ylabel("Cantidad de registros")
        plt.tight_layout()
        plt.show()

        # ---------- Barras: distribución PREDICHA ----------
        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=dist_pred.index,
            y=dist_pred.values
        )
        plt.title("Distribución predicha de clases de rotación (platos top)")
        plt.xlabel("Clase de rotación (predicha)")
        plt.ylabel("Cantidad de registros")
        plt.tight_layout()
        plt.show()

# ======================================================================================
# CORRELACIONES GLOBALES (usando predicciones_modelo_unico.csv si existe)
# ======================================================================================
if os.path.exists(PRED_ALL_PATH):
    print("\n" + "=" * 80)
    print("CORRELACIONES ENTRE KPIs Y VARIABLES OPERATIVAS")
    print("=" * 80)

    df_all = pd.read_csv(PRED_ALL_PATH)
    print(f"Filas en predicciones_modelo_unico.csv: {len(df_all)}")

    # Seleccionamos sólo las columnas numéricas relevantes que existan
    posibles = [
        "margen_plato_real",
        "ticket_dia_real",
        "cant_top_mes",
        "num_clientes",
        "ventas_totales_dia",
        "total_ticket",
        "margen_total_dia",
    ]
    corr_cols = [c for c in posibles if c in df_all.columns]

    if len(corr_cols) >= 2:
        corr = df_all[corr_cols].corr()
        print("\nMatriz de correlación (KPIs vs variables operativas):")
        print(corr.round(3))

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlación – KPIs y variables operativas")
        plt.tight_layout()
        plt.show()
    else:
        print("[CORR] No hay suficientes columnas numéricas para calcular correlaciones.")
else:
    print("\n[AVISO] No se encontró predicciones_modelo_unico.csv. Se omite sección de correlaciones.")

# ======================================================================================
# INTERPRETACIÓN AUTOMÁTICA (texto para exposición)
# ======================================================================================
print("\n" + "=" * 80)
print("INTERPRETACIÓN AUTOMATIZADA PARA EXPOSICIÓN")
print("=" * 80)

print(
"""
1️:Margen de ganancia por plato (KPI 1)
   • Se analiza el margen de cada línea de venta (plato en un ticket), comparando el margen
     REAL con el margen PREDICHO por el árbol de decisión.
   • Los TOP 10 casos positivos muestran platos/fechas donde se está ganando más de lo esperado;
     esto ayuda a identificar platos estrella o buenas combinaciones de precio y costo.
   • Los casos negativos muestran oportunidades de mejora: revisar recetas, porciones o precios.

2️:Ticket promedio diario (KPI 2)
   • El modelo estima el ticket promedio por día y modalidad (local / pensión).
   • Al comparar el ticket REAL con el ticket PREDICHO se identifican días “débiles” donde
     conviene lanzar promociones, combos o campañas, y días “fuertes” donde se debe asegurar
     suficiente stock y personal.
   • La distribución del ticket permite ver si el restaurante trabaja en un rango estable o muy
     variable de gasto por cliente.

3️:Rotación de platos top (KPI 3)
   • Para cada mes se toma el plato más vendido y se le asigna una clase de rotación
     (baja, media, alta) según la cantidad vendida.
   • El modelo aprende a clasificar esa rotación y se mide la exactitud entre la clase REAL y
     la PREDICHA. Esto es útil para anticipar qué platos serán “rápidos” y cuáles se moverán poco.
   • Con esta información se pueden ajustar compras, almacenamiento y planificación de menús.

-> En conjunto, el modelo unificado de árbol de decisión permite conectar:
   • La rentabilidad por plato (KPI1),
   • El comportamiento de consumo diario (KPI2),
   • Y la rotación de los platos clave a nivel mensual (KPI3),
   ofreciendo una base sólida para construir un dashboard web de apoyo a decisiones
   comerciales, operativas y de abastecimiento en el restaurante.
"""
)

print("\nFIN DEL REPORTE\n")