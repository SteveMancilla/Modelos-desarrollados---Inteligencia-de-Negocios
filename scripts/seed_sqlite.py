# scripts/seed_sqlite.py
import os, sqlite3, random
from datetime import datetime, timedelta, date
import numpy as np

# ---------- Paths / seeds ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "db", "talita.db")
os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)
random.seed(42); np.random.seed(42)

# ============================================================
# 1) CATÁLOGO DE PLATOS (30) + BEBIDAS EXTRA (sólo LOCAL)
# ============================================================
PLATOS = [
    "arroz_con_pollo","frejol_con_pescado","chicharron_cerdo","lomo_saltado","alberjita_guiso",
    "trucha_frita","aji_de_gallina","tallarin_saltado","chaufa_pollo","milanesa_pollo",
    "estofado_res","ensalada_rusa","tortilla_verduras","arroz_a_la_cubana","churrasco_res",
    "pollo_broaster","carapulcra","chanfainita","seco_de_res","seco_de_pollo",
    "tallarin_verde","adobo_cerdo","cau_cau","arroz_con_mariscos","bistec_a_lo_pobre",
    "salchipapa","escabeche_pescado","higado_encebollado","sudado_pescado","fetuccini_alfredo"
]

BEBIDAS_EXTRA = ["gaseosa_500ml","gaseosa_1L","sporade_500ml","jugo_frugos_473ml","agua_500ml"]

PRECIO_VENTA_PLATO_MIN = 10.0
PRECIO_VENTA_PLATO_MAX = 15.0
PRECIOS_BEBIDAS = {
    "gaseosa_500ml": (2.5, 3.0),
    "gaseosa_1L":    (5.0, 7.0),
    "sporade_500ml": (2.5, 3.5),
    "jugo_frugos_473ml": (2.5, 3.5),
    "agua_500ml":    (1.5, 2.5),
}

# ============================================================
# 2) INSUMOS DETALLADOS (unidad + costo unitario base)
# ============================================================
INSUMOS = [
    # Proteínas / mariscos
    ("pollo_entero","kg",12.0),("pechuga_pollo","kg",18.0),("carne_res","kg",24.0),
    ("higado_res","kg",16.0),("cerdo_corte","kg",20.0),("pescado_blanco","kg",26.0),
    ("trucha_fresca","kg",28.0),("mariscos_mix","kg",35.0),

    # Carbohidratos / pastas
    ("arroz_grano","kg",3.2),("papa_blanca","kg",1.8),("papa_amarilla","kg",2.3),
    ("papa_seca","kg",7.0),("fideo_spaghetti","kg",6.5),("fideo_tallarin","kg",6.5),
    ("harina_trigo","kg",4.0),("platano","kg",4.5),

    # Legumbres
    ("frejol_canario","kg",6.0),

    # Verduras / hortalizas
    ("cebolla","kg",2.4),("tomate","kg",3.0),("pimiento","kg",4.0),("zanahoria","kg",2.2),
    ("arveja","kg",5.2),("lechuga","kg",2.8),("limon","kg",6.5),

    # Aceites / lácteos / varios
    ("aceite_vegetal","lt",8.5),("leche","lt",5.0),("mantequilla","kg",22.0),
    ("queso_parmesano","kg",40.0),("huevo","unidad",0.7),("pan_hamburguesa","unidad",0.8),

    # Condimentos / salsas
    ("ajo","kg",8.0),("comino","kg",18.0),("pimienta","kg",24.0),("sal","kg",1.5),
    ("palillo_curcuma","kg",28.0),("aji_panca_pasta","kg",22.0),("aji_amarillo_pasta","kg",22.0),
    ("oregano_seco","kg",28.0),("laurel_hoja","kg",40.0),("vinagre","lt",6.0),
    ("sillao","lt",18.0),("azucar","kg",4.2),

    # Bebidas adicionales (para local)
    ("coca_500ml_unid","unidad",1.8),("sporade_500ml_unid","unidad",1.9),
    ("frugos_473ml_unid","unidad",1.8),("agua_500ml_unid","unidad",1.2),

    # Agua para refresco incluido
    ("agua","lt",1.0),
]

def unit_price(insumo_nombre):
    for n,u,c in INSUMOS:
        if n == insumo_nombre: return c,u
    raise ValueError(f"Insumo no encontrado en catálogo: {insumo_nombre}")

def cost_for(insumo_nombre, cantidad, unidad):
    costo, u_base = unit_price(insumo_nombre)
    if u_base == "kg":
        mult = 1.0 if unidad=="kg" else (1/1000.0 if unidad=="g" else None)
    elif u_base == "lt":
        mult = 1.0 if unidad=="lt" else (1/1000.0 if unidad=="ml" else None)
    elif u_base == "unidad":
        mult = 1.0 if unidad=="unidad" else None
    else:
        mult = None
    if mult is None:
        raise ValueError(f"Unidad incompatible para {insumo_nombre}: {unidad} (base {u_base})")
    return round(costo * cantidad * mult, 4)

# ============================================================
# 3) RECETAS por porción (+ refresco incluido)
# ============================================================
RECETAS = {
    "arroz_con_pollo":[("arroz_grano","g",180),("pollo_entero","g",160),("cebolla","g",20),("ajo","g",5),("arveja","g",20),("aceite_vegetal","ml",10),("sal","g",3),("comino","g",1)],
    "frejol_con_pescado":[("frejol_canario","g",160),("pescado_blanco","g",140),("cebolla","g",20),("ajo","g",5),("aceite_vegetal","ml",8),("sal","g",3)],
    "chicharron_cerdo":[("cerdo_corte","g",180),("ajo","g",5),("sal","g",3),("papa_blanca","g",160),("aceite_vegetal","ml",12)],
    "lomo_saltado":[("carne_res","g",160),("cebolla","g",40),("tomate","g",40),("papa_blanca","g",160),("aceite_vegetal","ml",12),("sillao","ml",8),("sal","g",3),("pimienta","g",1)],
    "alberjita_guiso":[("arveja","g",120),("cebolla","g",30),("ajo","g",5),("papa_blanca","g",120),("aceite_vegetal","ml",10),("sal","g",3)],
    "trucha_frita":[("trucha_fresca","g",180),("aceite_vegetal","ml",14),("papa_blanca","g",160),("sal","g",3),("limon","ml",10)],
    "aji_de_gallina":[("pechuga_pollo","g",140),("aji_amarillo_pasta","g",15),("leche","ml",40),("pan_hamburguesa","unidad",1),("queso_parmesano","g",5),("ajo","g",5),("aceite_vegetal","ml",8),("sal","g",3)],
    "tallarin_saltado":[("fideo_tallarin","g",160),("carne_res","g",120),("cebolla","g",30),("tomate","g",30),("aceite_vegetal","ml",10),("sillao","ml",8),("sal","g",3)],
    "chaufa_pollo":[("arroz_grano","g",180),("pechuga_pollo","g",120),("cebolla","g",20),("huevo","unidad",1),("aceite_vegetal","ml",10),("sillao","ml",8),("sal","g",3)],
    "milanesa_pollo":[("pechuga_pollo","g",160),("harina_trigo","g",20),("huevo","unidad",1),("aceite_vegetal","ml",12),("papa_blanca","g",140),("sal","g",3)],
    "estofado_res":[("carne_res","g",160),("cebolla","g",30),("zanahoria","g",30),("papa_amarilla","g",140),("aceite_vegetal","ml",8),("laurel_hoja","g",0.5),("sal","g",3)],
    "ensalada_rusa":[("papa_amarilla","g",160),("zanahoria","g",40),("arveja","g",30),("huevo","unidad",1),("sal","g",3)],
    "tortilla_verduras":[("huevo","unidad",2),("cebolla","g",20),("tomate","g",20),("aceite_vegetal","ml",8),("sal","g",3)],
    "arroz_a_la_cubana":[("arroz_grano","g",180),("huevo","unidad",2),("platano","g",100)],
    "churrasco_res":[("carne_res","g",180),("papa_blanca","g",160),("sal","g",3),("aceite_vegetal","ml",10)],
    "pollo_broaster":[("pollo_entero","g",180),("harina_trigo","g",20),("aceite_vegetal","ml",16),("sal","g",3)],
    "carapulcra":[("cerdo_corte","g",120),("papa_seca","g",60),("aji_panca_pasta","g",12),("aceite_vegetal","ml",10),("sal","g",3)],
    "chanfainita":[("higado_res","g",160),("papa_amarilla","g",140),("aji_panca_pasta","g",10),("ajo","g",5),("sal","g",3)],
    "seco_de_res":[("carne_res","g",160),("arveja","g",30),("cebolla","g",30),("ajo","g",5),("aceite_vegetal","ml",10),("sal","g",3)],
    "seco_de_pollo":[("pechuga_pollo","g",160),("arveja","g",30),("cebolla","g",30),("ajo","g",5),("aceite_vegetal","ml",10),("sal","g",3)],
    "tallarin_verde":[("fideo_spaghetti","g",180),("leche","ml",40),("aceite_vegetal","ml",10),("queso_parmesano","g",6),("sal","g",3)],
    "adobo_cerdo":[("cerdo_corte","g",180),("vinagre","ml",10),("ajo","g",5),("aji_panca_pasta","g",10),("sal","g",3)],
    "cau_cau":[("higado_res","g",140),("papa_amarilla","g",140),("palillo_curcuma","g",2),("cebolla","g",30),("ajo","g",5),("sal","g",3)],
    "arroz_con_mariscos":[("arroz_grano","g",180),("mariscos_mix","g",150),("ajo","g",5),("pimiento","g",20),("aceite_vegetal","ml",12),("sal","g",3)],
    "bistec_a_lo_pobre":[("carne_res","g",180),("papa_blanca","g",160),("huevo","unidad",1),("aceite_vegetal","ml",12),("sal","g",3)],
    "salchipapa":[("papa_blanca","g",240),("aceite_vegetal","ml",14),("sal","g",3)],
    "escabeche_pescado":[("pescado_blanco","g",160),("cebolla","g",40),("vinagre","ml",10),("aceite_vegetal","ml",10),("sal","g",3)],
    "higado_encebollado":[("higado_res","g",180),("cebolla","g",40),("aceite_vegetal","ml",10),("sal","g",3)],
    "sudado_pescado":[("pescado_blanco","g",180),("tomate","g",40),("cebolla","g",30),("ajo","g",5),("sal","g",3)],
    "fetuccini_alfredo":[("fideo_spaghetti","g",180),("leche","ml",40),("mantequilla","g",12),("queso_parmesano","g",8),("sal","g",3)],
}
# refresco incluido (limonada básica)
for plato in RECETAS:
    RECETAS[plato] += [("agua","ml",250),("azucar","g",10),("limon","ml",20)]

# ============================================================
# 4) ESQUEMA SQL
# ============================================================
conn = sqlite3.connect(DB_PATH); cur = conn.cursor()
cur.executescript("""
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS ventas;
DROP TABLE IF EXISTS recetas_detalle;
DROP TABLE IF EXISTS platos;
DROP TABLE IF EXISTS insumos;
DROP TABLE IF EXISTS inventario_diario;

CREATE TABLE platos(
  id_plato INTEGER PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL
);

CREATE TABLE insumos(
  id_insumo INTEGER PRIMARY KEY,
  nombre TEXT UNIQUE NOT NULL,
  unidad TEXT NOT NULL,
  costo_unitario REAL NOT NULL
);

CREATE TABLE recetas_detalle(
  id INTEGER PRIMARY KEY,
  plato_id INTEGER NOT NULL,
  insumo_id INTEGER NOT NULL,
  cantidad REAL NOT NULL,
  unidad TEXT NOT NULL,
  FOREIGN KEY(plato_id) REFERENCES platos(id_plato),
  FOREIGN KEY(insumo_id) REFERENCES insumos(id_insumo)
);

CREATE TABLE ventas(
  id_venta INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_id INTEGER NOT NULL,
  fecha TEXT NOT NULL,
  dia_semana TEXT NOT NULL,
  modalidad TEXT NOT NULL,
  item TEXT NOT NULL,
  cantidad INTEGER NOT NULL,
  precio_unitario REAL NOT NULL,
  costo_unitario REAL NOT NULL,
  promocion INTEGER NOT NULL,
  monto_linea REAL NOT NULL,
  costo_linea REAL NOT NULL,
  margen_ganancia REAL NOT NULL
);

CREATE TABLE inventario_diario(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fecha TEXT NOT NULL,
  insumo TEXT NOT NULL,
  categoria TEXT NOT NULL,
  costo_unitario REAL NOT NULL,
  stock_inicio REAL NOT NULL,
  ingreso REAL NOT NULL,
  consumo REAL NOT NULL,
  stock_fin REAL NOT NULL
);
""")
conn.commit()

# ============================================================
# 5) CARGA CATÁLOGOS + VALIDACIÓN
# ============================================================
cur.executemany("INSERT INTO platos(nombre) VALUES (?)", [(p,) for p in PLATOS])
cur.executemany("INSERT INTO insumos(nombre, unidad, costo_unitario) VALUES (?,?,?)", INSUMOS)
conn.commit()

# Validación de recetas
catalogo = {n for (n,_,_) in INSUMOS}
faltantes = set()
for plato, items in RECETAS.items():
    for ins,u,cant in items:
        if ins not in catalogo: faltantes.add(ins)
if faltantes:
    raise ValueError(f"Insumos faltantes en catálogo: {sorted(faltantes)}")

def get_insumo(nombre):
    r = cur.execute("SELECT id_insumo, unidad, costo_unitario FROM insumos WHERE nombre=?",(nombre,)).fetchone()
    if not r: raise ValueError(f"Insumo no encontrado: {nombre}")
    return r

def get_plato(nombre):
    r = cur.execute("SELECT id_plato FROM platos WHERE nombre=?",(nombre,)).fetchone()
    if not r: raise ValueError(f"Plato no encontrado: {nombre}")
    return r[0]

for plato, items in RECETAS.items():
    pid = get_plato(plato)
    for ins,u,cant in items:
        iid,_,_ = get_insumo(ins)
        cur.execute("INSERT INTO recetas_detalle(plato_id, insumo_id, cantidad, unidad) VALUES (?,?,?,?)",
                    (pid,iid,float(cant),u))
conn.commit()

# ============================================================
# 6) Costeo por porción (desde receta)
# ============================================================
def costo_porcion_plato(plato_nombre):
    pid = get_plato(plato_nombre)
    filas = cur.execute("""
        SELECT i.nombre, i.unidad, i.costo_unitario, r.cantidad, r.unidad
        FROM recetas_detalle r
        JOIN insumos i ON i.id_insumo = r.insumo_id
        WHERE r.plato_id = ?
    """,(pid,)).fetchall()
    total = 0.0
    for nom, u_base, c_base, cant, u_rec in filas:
        conv = "g" if (nom=="limon" and u_base=="kg" and u_rec=="ml") else u_rec
        total += cost_for(nom, cant, conv)
    return round(total,4)

# ============================================================
# 7) VENTAS (2023-01-01 a HOY) con menos ventas en 2025
#    y tope de líneas en ventas (~2500)
# ============================================================
def dia_str(dt): return dt.strftime("%a")

START = date(2023,1,1)
END   = date.today()
days  = (END - START).days + 1

rows = []
ticket_seq = 1

BASE_WEEKDAY = 16
BASE_WEEKEND = 22
YEAR_FACTOR = {2023:1.00, 2024:0.90, 2025:0.60}
TARGET_MAX_LINEAS = 2500

for i in range(days):
    if len(rows) >= TARGET_MAX_LINEAS: break
    fecha = START + timedelta(days=i)
    y = fecha.year
    weekday = fecha.weekday()  # 0=Mon ... 6=Sun
    base = BASE_WEEKEND if weekday in [4,5,6] else BASE_WEEKDAY
    lam = max(1, int(base * YEAR_FACTOR.get(y, 0.8)))
    tickets_dia = np.random.poisson(lam)

    for _ in range(max(1, tickets_dia)):
        if len(rows) >= TARGET_MAX_LINEAS: break
        ticket_id = ticket_seq; ticket_seq += 1
        modalidad = "pension" if np.random.rand() < 0.35 else "local"

        # ---------------------- PLATO (precio entero) ----------------------
        plato = np.random.choice(PLATOS)

        # precio inicial aleatorio -> redondeado a ENTERO
        precio_plato = int(round(np.random.uniform(PRECIO_VENTA_PLATO_MIN, PRECIO_VENTA_PLATO_MAX)))

        # costo por porción (float) y corrección si el costo >= precio
        costo_plato  = costo_porcion_plato(plato)
        if costo_plato >= precio_plato:
            # asegurar al menos 1 sol por encima del costo, en ENTERO
            precio_plato = int(np.ceil(costo_plato + np.random.uniform(1.0, 2.0)))

        # promo 10% (si aplica) y se vuelve a redondear a ENTERO
        promo = 1 if (modalidad=="local" and weekday in [0,1] and np.random.rand()<0.18) else 0
        if promo:
            precio_plato = max(int(PRECIO_VENTA_PLATO_MIN), int(round(precio_plato * 0.90)))

        monto = float(precio_plato * 1)          # ENTERO en soles, pero guardado como numérico
        costo = round(costo_plato * 1, 2)        # costo sigue siendo flotante
        margen = round(monto - costo, 2)

        rows.append((
            ticket_id, fecha.isoformat(), dia_str(datetime.combine(fecha, datetime.min.time())),
            modalidad, plato, 1, precio_plato, costo_plato, promo, monto, costo, margen
        ))

        # ---------------------- BEBIDAS (precios con decimales) ----------------------
        if modalidad == "local":
            n_beb = np.random.choice([0,1,2], p=[0.75,0.2,0.05])
            for _ in range(n_beb):
                if len(rows) >= TARGET_MAX_LINEAS: break
                bebida = np.random.choice(BEBIDAS_EXTRA)
                pmin,pmax = PRECIOS_BEBIDAS[bebida]
                precio_beb = round(np.random.uniform(pmin,pmax),2)  # bebidas se mantienen flotantes
                ins_map = {
                    "gaseosa_500ml":"coca_500ml_unid",
                    "gaseosa_1L":"coca_500ml_unid",   # ~2x 500ml
                    "sporade_500ml":"sporade_500ml_unid",
                    "jugo_frugos_473ml":"frugos_473ml_unid",
                    "agua_500ml":"agua_500ml_unid"
                }
                ins_beb = ins_map[bebida]
                _,_,costo_unit = get_insumo(ins_beb)
                costo_beb = costo_unit * (2.0 if bebida=="gaseosa_1L" else 1.0)
                monto_b = round(precio_beb * 1, 2)
                costo_b = round(costo_beb * 1, 2)
                margen_b = round(monto_b - costo_b, 2)

                rows.append((
                    ticket_id, fecha.isoformat(), dia_str(datetime.combine(fecha, datetime.min.time())),
                    modalidad, bebida, 1, precio_beb, costo_beb, 0, monto_b, costo_b, margen_b
                ))

cur.executemany("""
INSERT INTO ventas
(ticket_id, fecha, dia_semana, modalidad, item, cantidad, precio_unitario, costo_unitario,
 promocion, monto_linea, costo_linea, margen_ganancia)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
""", rows)
conn.commit()

# ============================================================
# 8) INVENTARIO semanal (para no crecer demasiado)
# ============================================================
CATEG = {
    "pollo":"proteina","pechuga":"proteina","carne":"proteina","higado":"proteina",
    "cerdo":"proteina","pescado":"proteina","trucha":"proteina","mariscos":"proteina",
    "arroz":"carbohidrato","papa":"carbohidrato","fideo":"carbohidrato","harina":"carbohidrato",
    "cebolla":"verdura","tomate":"verdura","pimiento":"verdura","zanahoria":"verdura",
    "arveja":"verdura","lechuga":"verdura","limon":"verdura",
    "aceite":"insumo_base","leche":"insumo_base","huevo":"insumo_base",
    "pan":"insumo_base","queso":"insumo_base","mantequilla":"insumo_base","azucar":"insumo_base",
    "ajo":"condimento","comino":"condimento","pimienta":"condimento","sal":"condimento",
    "palillo":"condimento","aji":"condimento","oregano":"condimento","laurel":"condimento",
    "vinagre":"condimento","sillao":"condimento",
    "coca_500ml_unid":"bebida","sporade_500ml_unid":"bebida","frugos_473ml_unid":"bebida","agua_500ml_unid":"bebida"
}
def guess_cat(n):
    for k,v in CATEG.items():
        if k in n: return v
    return "otros"

inv_rows = []
WEEK_STEP = 7
for i in range(0, days, WEEK_STEP):
    fecha = START + timedelta(days=i)
    for nom, uni, cu in INSUMOS:
        base = 30 if guess_cat(nom)!="bebida" else 45
        stock_ini = max(0, np.random.normal(base, base*0.22))
        ingreso   = max(0, np.random.normal(10 if guess_cat(nom)!="bebida" else 18, 4))
        consumo   = max(0, np.random.normal(12 if guess_cat(nom)!="bebida" else 16, 5))
        stock_fin = max(0, stock_ini + ingreso - consumo)
        inv_rows.append((
            fecha.isoformat(), nom, guess_cat(nom), round(cu,2),
            round(stock_ini,2), round(ingreso,2), round(consumo,2), round(stock_fin,2)
        ))

cur.executemany("""
INSERT INTO inventario_diario
(fecha, insumo, categoria, costo_unitario, stock_inicio, ingreso, consumo, stock_fin)
VALUES (?,?,?,?,?,?,?,?)
""", inv_rows)
conn.commit()

# ============================================================
# 9) Resumen
# ============================================================
n_lineas  = cur.execute("SELECT COUNT(*) FROM ventas").fetchone()[0]
n_tickets = cur.execute("SELECT COUNT(DISTINCT ticket_id) FROM ventas").fetchone()[0]
n_platos  = cur.execute("SELECT COUNT(*) FROM platos").fetchone()[0]
n_insumos = cur.execute("SELECT COUNT(*) FROM insumos").fetchone()[0]
print(f"OK -> lineas ventas: {n_lineas} | tickets: {n_tickets} | platos: {n_platos} | insumos: {n_insumos} | DB: {DB_PATH}")

conn.close()