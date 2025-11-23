# models/modelo_unico/report_modelo_unico_console.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")
sns.set()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

KPI1_PATH = os.path.join(THIS_DIR, "resultados_kpi1_margen_plato.csv")
KPI2_PATH = os.path.join(THIS_DIR, "resultados_kpi2_ticket_diario.csv")
KPI3_PATH = os.path.join(THIS_DIR, "resultados_kpi3_rotacion.csv")


# =====================================================================
# Helper para elegir columnas aunque el nombre cambie un poco
# =====================================================================
def pick_col(df: pd.DataFrame, candidates, kpi_name, logical_name):
    """
    Devuelve la primera columna encontrada en 'candidates'.
    Lanza error legible si ninguna existe.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"[{kpi_name}] No se encontró ninguna columna {candidates} "
        f"para '{logical_name}'. Revisa el CSV o ajusta la lista de candidatos."
    )


# =====================================================================
# 1) KPI-1: MARGEN DE GANANCIA POR PLATO
# =====================================================================
def reporte_kpi1():
    if not os.path.exists(KPI1_PATH):
        print(f"\n❌ No se encontró {KPI1_PATH}. Ejecuta primero train_arbol_multioutput.py")
        return

    df = pd.read_csv(KPI1_PATH)

    print("\n" + "=" * 80)
    print("KPI-1: MARGEN DE GANANCIA POR PLATO")
    print("=" * 80)
    print(f"Filas en resultados_kpi1_margen_plato: {len(df)}")
    print("Columnas:", list(df.columns))

    # -------- Mapeo flexible de columnas --------
    col_fecha = pick_col(
        df,
        ["fecha"],
        "KPI1",
        "fecha",
    )
    col_modalidad = pick_col(
        df,
        ["modalidad"],
        "KPI1",
        "modalidad",
    )
    col_plato = pick_col(
        df,
        ["nombre_plato", "plato", "item"],
        "KPI1",
        "nombre del plato",
    )
    col_margen_real = pick_col(
        df,
        ["margen_plato_real", "margen_real", "margen_total_dia"],
        "KPI1",
        "margen real",
    )
    col_margen_pred = pick_col(
        df,
        ["margen_plato_pred", "margen_pred", "margen_predicho"],
        "KPI1",
        "margen predicho",
    )
    col_precio = pick_col(
        df,
        ["precio_plato", "precio_unitario", "precio"],
        "KPI1",
        "precio del plato",
    )
    col_costo = pick_col(
        df,
        ["costo_plato", "costo_unitario", "costo"],
        "KPI1",
        "costo del plato",
    )

    # Aseguramos numéricos
    for c in [col_margen_real, col_margen_pred, col_precio, col_costo]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Residual
    df["res_margen"] = df[col_margen_real] - df[col_margen_pred]

    # -------- Resumen estadístico global --------
    stats = df[[col_margen_real, col_margen_pred, "res_margen"]].describe()
    print("\nResumen estadístico margen por plato (real vs predicho):")
    print(stats)

    # -------- Top 10 platos con mayor margen real promedio --------
    agg_plato = (
        df.groupby(col_plato)[[col_margen_real, col_margen_pred, "res_margen"]]
        .mean()
        .sort_values(col_margen_real, ascending=False)
        .head(10)
        .round(2)
        .reset_index()
    )

    print("\nTOP 10 platos más rentables (promedio de margen real):")
    print(agg_plato.to_string(index=False))

    # -------- Platos con peor desempeño (margen real promedio bajo) --------
    worst_plato = (
        df.groupby(col_plato)[[col_margen_real, col_margen_pred, "res_margen"]]
        .mean()
        .sort_values(col_margen_real, ascending=True)
        .head(10)
        .round(2)
        .reset_index()
    )

    print("\nTOP 10 platos con menor margen real promedio:")
    print(worst_plato.to_string(index=False))

    # -------- Scatter real vs predicho --------
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[col_margen_real], y=df[col_margen_pred], alpha=0.6)
    mmin = np.nanmin(df[col_margen_real])
    mmax = np.nanmax(df[col_margen_real])
    plt.plot([mmin, mmax], [mmin, mmax], "r--")
    plt.title("Margen real vs margen predicho (por plato)")
    plt.xlabel("Margen real (S/)")
    plt.ylabel("Margen predicho (S/)")
    plt.tight_layout()
    plt.show()

    # -------- Histograma de margen real --------
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col_margen_real], kde=True)
    plt.title("Distribución del margen real por plato")
    plt.xlabel("Margen real (S/)")
    plt.tight_layout()
    plt.show()

    # -------- Barras: top platos por margen real promedio --------
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=agg_plato,
        x=col_margen_real,
        y=col_plato,
    )
    plt.title("Top 10 platos más rentables (margen real promedio)")
    plt.xlabel("Margen real promedio (S/)")
    plt.ylabel("Plato")
    plt.tight_layout()
    plt.show()

    # -------- Comentario automático --------
    print(
        "\nComentario automático KPI1:\n"
        "   • La tabla de TOP 10 muestra los platos que más margen aportan al negocio.\n"
        "   • La comparación real vs predicho permite ver si el modelo está\n"
        "     sobreestimando o subestimando la rentabilidad de algunos platos.\n"
        "   • Estos resultados son útiles para ajustar precios, porciones o promociones.\n"
    )


# =====================================================================
# 2) KPI-2: TICKET PROMEDIO DIARIO
# =====================================================================
def reporte_kpi2():
    if not os.path.exists(KPI2_PATH):
        print(f"\n❌ No se encontró {KPI2_PATH}. Ejecuta primero train_arbol_multioutput.py")
        return

    df = pd.read_csv(KPI2_PATH)

    print("\n" + "=" * 80)
    print("KPI-2: TICKET PROMEDIO DIARIO")
    print("=" * 80)
    print(f"Filas en resultados_kpi2_ticket_diario: {len(df)}")
    print("Columnas:", list(df.columns))

    col_fecha = pick_col(df, ["fecha"], "KPI2", "fecha")
    col_modalidad = pick_col(df, ["modalidad"], "KPI2", "modalidad")

    # Ajusta estas listas si en tu CSV se llaman distinto
    col_ticket_real = pick_col(
        df,
        ["ticket_real", "ticket_promedio_dia", "ticket"],
        "KPI2",
        "ticket promedio real",
    )
    col_ticket_pred = pick_col(
        df,
        ["ticket_pred", "ticket_predicho"],
        "KPI2",
        "ticket promedio predicho",
    )
    col_num_clientes = pick_col(
        df,
        ["num_clientes", "clientes_dia"],
        "KPI2",
        "número de clientes",
    )
    col_ventas = pick_col(
        df,
        ["ventas_totales_dia", "ventas_totales", "total_ventas"],
        "KPI2",
        "ventas totales del día",
    )

    for c in [col_ticket_real, col_ticket_pred, col_num_clientes, col_ventas]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["res_ticket"] = df[col_ticket_real] - df[col_ticket_pred]

    # Resumen estadístico
    stats = df[[col_ticket_real, col_ticket_pred, "res_ticket"]].describe()
    print("\nResumen estadístico del ticket promedio diario:")
    print(stats)

    # Días con ticket mucho más bajo de lo esperado
    dias_bajos = (
        df.sort_values("res_ticket")
        .head(10)[[col_fecha, col_modalidad, col_ticket_real, col_ticket_pred, "res_ticket"]]
        .round(2)
    )
    print("\nTOP 10 días con ticket por debajo de lo esperado:")
    print(dias_bajos.to_string(index=False))

    # Días con ticket muy por encima de lo esperado
    dias_altos = (
        df.sort_values("res_ticket", ascending=False)
        .head(10)[[col_fecha, col_modalidad, col_ticket_real, col_ticket_pred, "res_ticket"]]
        .round(2)
    )
    print("\nTOP 10 días con ticket por encima de lo esperado:")
    print(dias_altos.to_string(index=False))

    # Scatter real vs predicho
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[col_ticket_real], y=df[col_ticket_pred], alpha=0.6)
    tmin = np.nanmin(df[col_ticket_real])
    tmax = np.nanmax(df[col_ticket_real])
    plt.plot([tmin, tmax], [tmin, tmax], "r--")
    plt.title("Ticket promedio real vs predicho (por día)")
    plt.xlabel("Ticket real (S/)")
    plt.ylabel("Ticket predicho (S/)")
    plt.tight_layout()
    plt.show()

    # Histograma ticket real
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col_ticket_real], kde=True)
    plt.title("Distribución del ticket promedio diario (real)")
    plt.xlabel("Ticket (S/)")
    plt.tight_layout()
    plt.show()

    # Correlación simple con número de clientes y ventas totales
    corr = df[[col_ticket_real, col_num_clientes, col_ventas]].corr().round(3)
    print("\nMatriz de correlación (ticket vs clientes y ventas):")
    print(corr)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlación: ticket promedio vs clientes y ventas")
    plt.tight_layout()
    plt.show()

    print(
        "\nComentario automático KPI2:\n"
        "   • Los días con ticket bajo son candidatos para aplicar promociones,\n"
        "     combos o actividades de marketing.\n"
        "   • Los días con ticket alto muestran momentos fuertes del negocio,\n"
        "     donde conviene asegurar stock y personal.\n"
        "   • La correlación con número de clientes y ventas totales ayuda a\n"
        "     entender si el ticket sube por volumen o por valor de consumo.\n"
    )


# =====================================================================
# 3) KPI-3: ROTACIÓN (PLATO TOP POR MES)
# =====================================================================
def reporte_kpi3():
    if not os.path.exists(KPI3_PATH):
        print(f"\n❌ No se encontró {KPI3_PATH}. Ejecuta primero train_arbol_multioutput.py")
        return

    df = pd.read_csv(KPI3_PATH)

    print("\n" + "=" * 80)
    print("KPI-3: ROTACIÓN – PLATOS MÁS VENDIDOS POR MES")
    print("=" * 80)
    print(f"Filas en resultados_kpi3_rotacion: {len(df)}")
    print("Columnas:", list(df.columns))

    col_mes = pick_col(df, ["mes"], "KPI3", "mes")
    col_plato_top = pick_col(df, ["plato_top_mes", "plato_top", "plato"], "KPI3", "plato top")
    col_cant = pick_col(df, ["cant_top_mes", "cantidad_vendida_mes", "cantidad"], "KPI3", "cantidad vendida mes")
    col_rot_pred = pick_col(df, ["rotacion_pred", "rotacion_code", "rotacion_clase"], "KPI3", "clase de rotación")

    df[col_cant] = pd.to_numeric(df[col_cant], errors="coerce")

    # Resumen por clase de rotación
    dist_rot = (
        df.groupby(col_rot_pred)[col_mes]
        .count()
        .reset_index(name="num_registros")
        .sort_values("num_registros", ascending=False)
    )

    print("\nDistribución de registros por clase de rotación predicha:")
    print(dist_rot.to_string(index=False))

    # TOP platos más vendidos (suma en todo el periodo)
    top_platos = (
        df.groupby(col_plato_top)[col_cant]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    print("\nTOP 10 platos más vendidos en todo el periodo (suma de unidades):")
    print(top_platos.to_string(index=False))

    # TOP meses con más rotación (suma de unidades del plato top)
    top_meses = (
        df.groupby(col_mes)[col_cant]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )

    print("\nTOP 12 meses con mayor volumen de venta del plato top:")
    print(top_meses.to_string(index=False))

    # Barras: clases de rotación
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col_rot_pred)
    plt.title("Distribución de clases de rotación predicha")
    plt.xlabel("Clase de rotación (0=baja,1=media,2=alta)")
    plt.ylabel("Cantidad de registros")
    plt.tight_layout()
    plt.show()

    # Barras: top platos
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top_platos, x=col_cant, y=col_plato_top)
    plt.title("Top 10 platos más vendidos (rotación)")
    plt.xlabel("Unidades vendidas (suma en todo el periodo)")
    plt.ylabel("Plato")
    plt.tight_layout()
    plt.show()

    # Línea: evolución mensual de la cantidad del plato top
    plt.figure(figsize=(9, 4))
    # Orden cronológico por mes
    df_mes = top_meses.copy()
    # Para ordenar correctamente por fecha si el formato es YYYY-MM
    df_mes["mes_orden"] = pd.to_datetime(df_mes[col_mes] + "-01", errors="coerce")
    df_mes = df_mes.sort_values("mes_orden")
    plt.plot(df_mes[col_mes], df_mes[col_cant], marker="o")
    plt.title("Evolución de la rotación (plato top por mes)")
    plt.xlabel("Mes")
    plt.ylabel("Unidades vendidas del plato top")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(
        "\nComentario automático KPI3:\n"
        "   • La clase de rotación predicha (0,1,2) indica qué tan rápido se\n"
        "     mueven los platos top en cada mes.\n"
        "   • Los platos con mayores unidades vendidas son candidatos para\n"
        "     asegurar stock y negociar mejores condiciones con proveedores.\n"
        "   • La curva de evolución mensual permite ver estacionalidad y\n"
        "     planificar compras e inventario con anticipación.\n"
    )


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REPORTE CONSOLIDADO – MODELO ÚNICO (3 KPI, CSVs separados)")
    print("=" * 80)

    # Cada función usa SU CSV correspondiente
    reporte_kpi1()
    reporte_kpi2()
    reporte_kpi3()

    print("\nFIN DEL REPORTE CONSOLIDADO.\n")