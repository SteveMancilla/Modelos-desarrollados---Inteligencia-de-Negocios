# models/kpi2_ticket/report_kpi2_console.py
import os
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = THIS_DIR

# Cargar CSVs
df_final  = pd.read_csv(os.path.join(OUT_DIR, "kpi2_resumen_final.csv"))
df_dias   = pd.read_csv(os.path.join(OUT_DIR, "kpi2_top_dias.csv"))
df_dias_g = pd.read_csv(os.path.join(OUT_DIR, "kpi2_top_dias_global.csv"))
df_fechas = pd.read_csv(os.path.join(OUT_DIR, "kpi2_top_fechas.csv"))
df_stab   = pd.read_csv(os.path.join(OUT_DIR, "kpi2_estabilidad_modalidad.csv"))
df_imp    = pd.read_csv(os.path.join(OUT_DIR, "importancias.csv"))

print("\n" + "="*80)
print("RESUMEN KPI-2: TICKET PROMEDIO DIARIO")
print("="*80)

inicio, fin = df_final["fecha"].min(), df_final["fecha"].max()
print(f"\nPeriodo analizado: {inicio} → {fin}")
print(f"Registros (fecha+modalidad): {len(df_final):,}")

# Promedios por modalidad
promedios = (
    df_final.groupby("modalidad")["ticket_promedio"]
    .agg(["mean","std","min","max"]).round(2).reset_index()
)
print("\nTicket promedio general por modalidad:")
print(promedios.to_string(index=False))

# Mejores días por modalidad
print("\nDías con mayor ticket promedio por modalidad:")
for mod in df_dias["modalidad"].unique():
    sub = df_dias[df_dias["modalidad"] == mod].head(3)
    print(f"\n {mod.upper()}:")
    for _, row in sub.iterrows():
        print(f"   - {row['dia_semana']:<3}: S/ {row['ticket_promedio']:.2f}")

# Mejores días a nivel GLOBAL (sin modalidad)
print("\n Día de la semana con mayor ticket (GLOBAL):")
for _, r in df_dias_g.head(7).iterrows():
    print(f"  - {r['dia_semana']:<3}: S/ {r['ticket_promedio']:.2f}")

# Top fechas (para campañas puntuales)
print("\n Top 10 FECHAS con ticket más alto (GLOBAL):")
for _, r in df_fechas.head(10).iterrows():
    print(f"  - {r['fecha']} ({r['dia_semana']}): "
          f"Ticket S/ {r['ticket_promedio']:.2f} | "
          f"Clientes {int(r['num_clientes'])} | "
          f"Ventas S/ {r['total_ventas_dia']:.2f}")

# Estabilidad
print("\n Estabilidad del ticket (menor CV% = más estable):")
stab_prom = df_stab.groupby("modalidad")["cv_%"].mean().round(2)
for mod, cv in stab_prom.items():
    print(f"  - {mod.capitalize():<8}: {cv:.2f}% (coef. de variación promedio)")

# Importancias del árbol
print("\n Variables que más influyen en el ticket (modelo):")
for _, row in df_imp.sort_values("importance", ascending=False).head(5).iterrows():
    print(f"  • {row['feature']:<25} → {row['importance']:.3f}")

# Distribución de clase de ticket
clasif = df_final["clase_ticket"].value_counts(normalize=True).mul(100).round(1)
print("\n Distribución de ticket diario:")
for cat, val in clasif.items():
    print(f"  - {cat.capitalize():<6}: {val:.1f}% de los días")

# Conclusión
cv_local  = stab_prom.get("local", np.nan)
cv_pension = stab_prom.get("pension", np.nan)
conclusion = "\n CONCLUSIÓN PARA ACCIÓN:\n"
if cv_pension < cv_local:
    conclusion += f"- Pensión es más estable (CV {cv_pension:.1f}% vs {cv_local:.1f}%).\n"
else:
    conclusion += f"- Local es más estable (CV {cv_local:.1f}% vs {cv_pension:.1f}%).\n"

best_global = df_dias_g.iloc[0]
conclusion += (
    f"- El mejor día global por ticket es **{best_global['dia_semana']}** "
    f"(S/ {best_global['ticket_promedio']:.2f}); enfocar combos/upsell y visibilidad.\n"
    "- Usar ‘top fechas’ para calendarizar campañas y reforzar oferta según afluencia.\n"
)
print(conclusion)