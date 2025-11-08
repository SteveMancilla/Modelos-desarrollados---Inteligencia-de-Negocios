# models/kpi3_rotacion/report_kpi3_console_mensual.py
import os
import argparse
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = THIS_DIR

PATH_MENSUAL  = os.path.join(OUT_DIR, "kpi3_rotacion_mensual.csv")
PATH_CLUSTERS = os.path.join(OUT_DIR, "kpi3_clusters_insumos_mensual.csv")
PATH_CENTROS  = os.path.join(OUT_DIR, "kpi3_centroides_mensual.csv")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Reporte mensual KPI-3: Rotación de inventario (K-Means sobre promedio mensual)"
    )
    ap.add_argument("--mes", type=str, default=None, help="Filtrar por mes (YYYY-MM), ej: 2024-08")
    ap.add_argument("--categoria", type=str, default=None, help="Filtrar por categoría (texto libre)")
    ap.add_argument("--top", type=int, default=10, help="Tamaño de listados TOP (default=10)")
    return ap.parse_args()

def sanitize_mensual(df_mensual: pd.DataFrame) -> pd.DataFrame:
    # Asegurar existencia de columnas clave
    for col in ["mes","insumo","rotacion_mensual","consumo_total","inventario_promedio","costo_unitario","unidad"]:
        if col not in df_mensual.columns:
            # crea si falta, para evitar KeyError (se completará con NaN)
            df_mensual[col] = np.nan

    # Limpiar "mes": descartar NaN y asegurar string 'YYYY-MM'
    df_mensual = df_mensual.dropna(subset=["mes"]).copy()
    df_mensual["mes"] = df_mensual["mes"].astype(str)

    # Coaccionar numéricos
    num_cols = ["rotacion_mensual","consumo_total","inventario_promedio","costo_unitario"]
    for c in num_cols:
        df_mensual[c] = pd.to_numeric(df_mensual[c], errors="coerce")

    # Categoría si no existe
    if "categoria" not in df_mensual.columns:
        df_mensual["categoria"] = "sin_categoria"

    return df_mensual

def sanitize_clusters(df_clusters: pd.DataFrame, df_mensual: pd.DataFrame) -> pd.DataFrame:
    # Asegurar columnas base
    for col in ["insumo","unidad","costo_unitario","rotacion_mensual_prom","consumo_total_prom","inventario_promedio","clase_rotacion"]:
        if col not in df_clusters.columns:
            df_clusters[col] = np.nan

    # Intentar adjuntar categoría desde mensual (última conocida por insumo)
    if "categoria" in df_mensual.columns:
        cat_map = (df_mensual.sort_values("mes")
                             .groupby("insumo")["categoria"].last())
        df_clusters = df_clusters.merge(cat_map.rename("categoria"), on="insumo", how="left")
    df_clusters["categoria"] = df_clusters["categoria"].fillna("sin_categoria")

    # Coaccionar numéricos
    for c in ["rotacion_mensual_prom","consumo_total_prom","inventario_promedio","costo_unitario"]:
        df_clusters[c] = pd.to_numeric(df_clusters[c], errors="coerce")

    return df_clusters

def imprimir_top_clase(df, clase, top=10, titulo="", asc=False):
    sub = df[df["clase_rotacion"] == clase].copy()
    if sub.empty:
        print(f"\n {titulo}\n (No hay insumos en clase {clase})")
        return
    sub = sub.sort_values("rotacion_mensual_prom", ascending=asc).head(top)
    print(f"\n {titulo}")
    for _, r in sub.iterrows():
        print(f" - {r['insumo']}: rot_mens_prom={r['rotacion_mensual_prom']:.2f} | "
              f"cons_mens_prom={r['consumo_total_prom']:.0f} | "
              f"inv_prom={r['inventario_promedio']:.1f} | "
              f"S/ {r['costo_unitario']:.2f}/{r.get('unidad','')}")

def main():
    args = parse_args()

    # Chequeo de archivos
    if not (os.path.exists(PATH_MENSUAL) and os.path.exists(PATH_CLUSTERS)):
        raise SystemExit(" No se encontraron CSV mensuales. Ejecuta primero train_kmeans.py (versión mensual).")

    # Carga + sanitización
    df_mensual  = pd.read_csv(PATH_MENSUAL)
    df_mensual  = sanitize_mensual(df_mensual)

    df_clusters = pd.read_csv(PATH_CLUSTERS)
    df_clusters = sanitize_clusters(df_clusters, df_mensual)

    if df_mensual.empty or df_clusters.empty:
        raise SystemExit(" Los datasets están vacíos tras sanitización. Revisa que train_kmeans_mensual haya generado datos.")

    # Encabezado
    periodo_inicio = df_mensual["mes"].min()
    periodo_fin    = df_mensual["mes"].max()
    total_insumos  = df_clusters["insumo"].nunique()

    print("\n" + "="*80)
    print(" RESUMEN KPI-3 (MENSUAL): Rotación de inventario (K-Means sobre promedio mensual)")
    print("="*80)
    print(f"\n Periodo de ventas: {periodo_inicio} → {periodo_fin}")
    print(f" Insumos distintos (promedio mensual): {total_insumos}\n")

    # Filtros de categoría
    if args.categoria:
        df_clusters = df_clusters[df_clusters["categoria"].astype(str).str.contains(args.categoria, case=False, na=False)]
        df_mens_sub = df_mensual[df_mensual["categoria"].astype(str).str.contains(args.categoria, case=False, na=False)].copy()
        print(f" Filtro categoría: {args.categoria}  → Insumos: {df_clusters['insumo'].nunique()}")
    else:
        df_mens_sub = df_mensual.copy()

    # Distribución por clase (promedio mensual)
    dist = (df_clusters.groupby("clase_rotacion")
                      .agg(insumos=("insumo","count"),
                           rot_mens_prom=("rotacion_mensual_prom","mean"),
                           cons_mens_prom=("consumo_total_prom","mean"))
                      .reset_index()
                      .sort_values("rot_mens_prom", ascending=False))
    dist["rot_mens_prom"]  = dist["rot_mens_prom"].round(2)
    dist["cons_mens_prom"] = dist["cons_mens_prom"].round(0)

    print("\n Distribución por clase (promedios mensuales):")
    print(dist.to_string(index=False))

    # TOP por clase
    imprimir_top_clase(df_clusters, "alta",  args.top, " TOP insumos de ALTA rotación mensual (reposición frecuente):", asc=False)
    imprimir_top_clase(df_clusters, "media", args.top, " Insumos de ROTACIÓN MEDIA mensual:", asc=False)
    imprimir_top_clase(df_clusters, "baja",  args.top, " Insumos de BAJA rotación mensual (revisar compra/lotes):", asc=True)

    # Detalle del mes puntual (opcional)
    if args.mes:
        print("\n" + "-"*80)
        print(f" DETALLE DEL MES: {args.mes}")
        df_mes = df_mens_sub[df_mens_sub["mes"].astype(str) == args.mes].copy()
        if df_mes.empty:
            print(" (No hay datos para ese mes con los filtros aplicados.)")
        else:
            df_mes = df_mes.sort_values("rotacion_mensual", ascending=False)
            top_mes = df_mes.head(args.top)
            print("\n TOP por rotación en el mes:")
            for _, r in top_mes.iterrows():
                print(f" - {r['insumo']}: rot_mens={r['rotacion_mensual']:.2f} | "
                      f"consumo={r['consumo_total']:.0f} | inv_prom={r['inventario_promedio']:.1f}")

            print("\n Estadísticos de rotación del mes:")
            print(df_mes["rotacion_mensual"].describe().to_string())

    # Desglose por categoría
    if "categoria" in df_clusters.columns:
        tabla_cat = (df_clusters.groupby(["clase_rotacion","categoria"])["insumo"]
                               .count().rename("items").reset_index()
                               .sort_values(["clase_rotacion","items"], ascending=[True,False]))
        print("\n Desglose por categoría (conteo de insumos por clase):")
        for clase in ["alta","media","baja"]:
            sub = tabla_cat[tabla_cat["clase_rotacion"] == clase]
            if not sub.empty:
                print(f"\n  {clase.upper()}:")
                for _, r in sub.iterrows():
                    print(f"   - {r['categoria']}: {r['items']}")

    # Reglas tácticas
    print("\n Reglas tácticas de abastecimiento (mensual):")
    print(" - ALTA : reabastecimiento semanal; stock de seguridad según demanda pico.")
    print(" - MEDIA: compra quincenal/semanal según menús; vigilar desviaciones.")
    print(" - BAJA : lotes pequeños; monitorear vencimientos; considerar sustitución/recetas que incrementen uso.")

if __name__ == "__main__":
    main()