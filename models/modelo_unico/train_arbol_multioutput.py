# models/modelo_unico/train_arbol_multioutput.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# 1) Rutas
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_DIR, "dataset_unificado.csv")
OUT_DIR = THIS_DIR

if not os.path.exists(DATA_PATH):
    raise SystemExit("❌ No existe dataset_unificado.csv. Ejecuta build_dataset_unificado.py")

df = pd.read_csv(DATA_PATH)
print("Filas del dataset:", len(df))
print("Columnas:", df.columns.tolist())

# ============================================================
# 2) TARGETS (KPIs)
#    KPI1: margen por plato
#    KPI2: ticket promedio diario
#    KPI3: rotación (baja/media/alta) -> 0/1/2
# ============================================================

# --- KPI1: margen por plato (línea de detalle) ---
# margen_plato = (precio - costo) * cantidad
df["margen_plato"] = (df["precio_plato"] - df["costo_plato"]) * df["cantidad"]
df["margen_plato"] = pd.to_numeric(df["margen_plato"], errors="coerce")

# --- KPI2: ticket promedio diario ---
df["ticket_promedio_dia"] = pd.to_numeric(df["ticket_promedio_dia"], errors="coerce")

# --- KPI3: rotación en 3 clases, a partir de cant_top_mes ---
# puedes ajustar estos umbrales si ves que quedan desbalanceadas
def clas_rot(cant):
    if cant <= 10:
        return 0  # baja
    if cant <= 20:
        return 1  # media
    return 2      # alta

df["rotacion_code"] = df["cant_top_mes"].apply(clas_rot)

# Eliminamos filas sin targets válidos
df = df.dropna(subset=["margen_plato", "ticket_promedio_dia", "rotacion_code"]).copy()

# Aseguramos tipo entero para la clase de rotación
df["rotacion_code"] = df["rotacion_code"].astype(int)

# ============================================================
# 3) VARIABLES PREDICTORAS (X)
# ============================================================

# Columnas numéricas que alimentarán al modelo
num_cols = [
    "edad",
    "precio_plato",
    "costo_plato",
    "cantidad",
    "subtotal",
    "total_ticket",
    "ticket_promedio_ticket",
    "num_clientes",
    "ventas_totales_dia",
    "cant_top_mes"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        raise SystemExit(f"❌ Falta la columna numérica requerida: {c}")

# Columnas categóricas a one-hot
cat_cols = ["modalidad", "tipo_cliente", "categoria_plato", "extra"]

for c in cat_cols:
    if c not in df.columns:
        raise SystemExit(f"❌ Falta la columna categórica requerida: {c}")

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(df[cat_cols])
X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))

# Matriz numérica
X_num = df[num_cols].reset_index(drop=True)

# Matriz final de entrada
X = pd.concat([X_num, X_cat], axis=1)

# Matriz de salidas (3 KPIs)
y = df[["margen_plato", "ticket_promedio_dia", "rotacion_code"]].values

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 4) TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 5) ENTRENAMIENTO DEL MODELO MULTI-SALIDA
# ============================================================
tree = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=10,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

# ============================================================
# 6) MÉTRICAS POR KPI
# ============================================================

# --- KPI1: margen por plato ---
yt_margen = y_test[:, 0]
yp_margen = y_pred[:, 0]

print("\n===== KPI 1: MARGEN DE GANANCIA POR PLATO =====")
print(f"R² margen:   {r2_score(yt_margen, yp_margen):.4f}")
print(f"RMSE margen: {np.sqrt(mean_squared_error(yt_margen, yp_margen)):.2f} soles")

# --- KPI2: ticket promedio diario ---
yt_ticket = y_test[:, 1]
yp_ticket = y_pred[:, 1]

print("\n===== KPI 2: TICKET PROMEDIO DIARIO =====")
print(f"R² ticket:   {r2_score(yt_ticket, yp_ticket):.4f}")
print(f"RMSE ticket: {np.sqrt(mean_squared_error(yt_ticket, yp_ticket)):.2f} soles")

# --- KPI3: rotación (0=baja,1=media,2=alta) ---
yt_rot = y_test[:, 2].astype(int)
yp_rot = np.round(y_pred[:, 2]).clip(0, 2).astype(int)

print("\n===== KPI 3: CLASE DE ROTACIÓN =====")
print(f"Accuracy rotación: {accuracy_score(yt_rot, yp_rot):.4f}")
print("Matriz de confusión (filas=real, columnas=predicho):")
print(confusion_matrix(yt_rot, yp_rot))

# ============================================================
# 7) EXPORTAR CSVs POR KPI (PARA DASHBOARD)
# ============================================================

# Recuperamos filas originales correspondientes al conjunto de test
pred = df.iloc[X_test.index].copy()

pred["margen_plato_real"] = yt_margen
pred["margen_plato_pred"] = yp_margen
pred["ticket_dia_real"]   = yt_ticket
pred["ticket_dia_pred"]   = yp_ticket
pred["rotacion_real"]     = yt_rot
pred["rotacion_pred"]     = yp_rot

# CSV general con todo
pred_path_full = os.path.join(OUT_DIR, "predicciones_modelo_unico.csv")
pred.to_csv(pred_path_full, index=False)

# --- CSV KPI1: margen por plato ---
kpi1_cols = [
    "fecha", "modalidad",
    "nombre_plato", "categoria_plato",
    "precio_plato", "costo_plato",
    "cantidad",
    "margen_plato_real", "margen_plato_pred"
]
kpi1_path = os.path.join(OUT_DIR, "resultados_kpi1_margen_plato.csv")
pred[kpi1_cols].to_csv(kpi1_path, index=False)

# --- CSV KPI2: ticket promedio diario ---
kpi2_cols = [
    "fecha", "modalidad",
    "num_clientes", "ventas_totales_dia",
    "ticket_promedio_dia", "ticket_dia_pred"
]
kpi2_path = os.path.join(OUT_DIR, "resultados_kpi2_ticket_diario.csv")
pred[kpi2_cols].drop_duplicates(subset=["fecha", "modalidad"]).to_csv(kpi2_path, index=False)

# --- CSV KPI3: rotación (por plato top del mes) ---
kpi3_cols = [
    "mes", "plato_top_mes", "cant_top_mes",
    "rotacion_real", "rotacion_pred"
]
kpi3_path = os.path.join(OUT_DIR, "resultados_kpi3_rotacion.csv")
pred[kpi3_cols].drop_duplicates(subset=["mes", "plato_top_mes"]).to_csv(kpi3_path, index=False)

print("\nCSV generados:")
print("  •", pred_path_full)
print("  •", kpi1_path)
print("  •", kpi2_path)
print("  •", kpi3_path)

# ============================================================
# 8) IMPORTANCIA DE VARIABLES Y REGLAS DEL ÁRBOL
# ============================================================
importancias = pd.DataFrame({
    "feature": X.columns,
    "importance": tree.feature_importances_
}).sort_values("importance", ascending=False)

imp_path = os.path.join(OUT_DIR, "importancias_modelo_unico.csv")
importancias.to_csv(imp_path, index=False)

rules = export_text(tree, feature_names=list(X.columns))
rules_path = os.path.join(OUT_DIR, "reglas_arbol_modelo_unico.txt")
with open(rules_path, "w", encoding="utf-8") as f:
    f.write(rules)

print("\nArchivos de importancia y reglas:")
print("  •", imp_path)
print("  •", rules_path)
print("\n✅ Entrenamiento y exportación completados.")