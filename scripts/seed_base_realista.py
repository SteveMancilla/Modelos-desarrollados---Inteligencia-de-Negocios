# scripts/seed_base_realista.py
import os, sqlite3, random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("es_ES")
random.seed(42); np.random.seed(42)

# ============================================================
# 1) PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "db", "talita_realista.db")
os.makedirs(os.path.join(BASE_DIR, "db"), exist_ok=True)

# ============================================================
# 2) CONEXIÓN Y TABLAS (ESQUEMA SIMPLIFICADO Y REALISTA)
# ============================================================
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS clientes;
DROP TABLE IF EXISTS platos;
DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS detalle_ticket;
DROP TABLE IF EXISTS rotacion_insumos;

CREATE TABLE clientes(
    id_cliente INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    edad INTEGER,
    telefono TEXT,
    tipo_cliente TEXT          -- 'empresa' | 'ocasional'
);

CREATE TABLE platos(
    id_plato INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    categoria TEXT,
    precio REAL,
    costo REAL
);

CREATE TABLE tickets(
    id_ticket INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT,
    id_cliente INTEGER,
    modalidad TEXT,            -- 'local' | 'pension'
    total REAL,
    ticket_promedio REAL,
    FOREIGN KEY(id_cliente) REFERENCES clientes(id_cliente)
);

CREATE TABLE detalle_ticket(
    id_detalle INTEGER PRIMARY KEY AUTOINCREMENT,
    id_ticket INTEGER,
    id_plato INTEGER,
    cantidad INTEGER,
    subtotal REAL,
    extra TEXT,                -- 'gaseosa' | 'postre' | 'nada'
    FOREIGN KEY(id_ticket) REFERENCES tickets(id_ticket),
    FOREIGN KEY(id_plato) REFERENCES platos(id_plato)
);

-- IMPORTANTE: aquí solo guardamos el agregado mensual.
-- La clasificación de rotación (alta/media/baja) se hará luego con ML.
CREATE TABLE rotacion_insumos(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mes TEXT,                              -- 'YYYY-MM'
    id_plato INTEGER,
    cantidad_vendida_mes INTEGER,
    FOREIGN KEY(id_plato) REFERENCES platos(id_plato)
);
""")
conn.commit()

# ============================================================
# 3) CLIENTES (400 CLIENTES REALISTAS)
# ============================================================
TIPO_CLIENTE = ["empresa", "ocasional"]
clientes = []

for _ in range(400):
    # Teléfono peruano de 9 dígitos que empieza en 9 (ej: 9XXXXXXXX)
    telefono = "9" + "".join(str(random.randint(0, 9)) for _ in range(8))
    clientes.append((
        fake.name(),
        random.randint(18, 70),
        telefono,
        random.choice(TIPO_CLIENTE)
    ))

cur.executemany(
    "INSERT INTO clientes(nombre,edad,telefono,tipo_cliente) VALUES (?,?,?,?)",
    clientes
)
conn.commit()

# ============================================================
# 4) CATÁLOGO DE PLATOS
#    Usamos los mismos nombres del script original (30 platos)
#    Asignamos categoría genérica y precios/costos coherentes.
# ============================================================
PLATOS_NOMBRES = [
    "arroz_con_pollo","frejol_con_pescado","chicharron_cerdo","lomo_saltado","alberjita_guiso",
    "trucha_frita","aji_de_gallina","tallarin_saltado","chaufa_pollo","milanesa_pollo",
    "estofado_res","ensalada_rusa","tortilla_verduras","arroz_a_la_cubana","churrasco_res",
    "pollo_broaster","carapulcra","chanfainita","seco_de_res","seco_de_pollo",
    "tallarin_verde","adobo_cerdo","cau_cau","arroz_con_mariscos","bistec_a_lo_pobre",
    "salchipapa","escabeche_pescado","higado_encebollado","sudado_pescado","fetuccini_alfredo"
]

def categoria_plato(nombre: str) -> str:
    if "pollo" in nombre:      return "pollo"
    if "res" in nombre or "bistec" in nombre or "lomo" in nombre or "churrasco" in nombre:
        return "carne"
    if "pescado" in nombre or "trucha" in nombre or "mariscos" in nombre:
        return "pescado"
    if "ensalada" in nombre:  return "ensalada"
    if "fetuccini" in nombre or "tallarin" in nombre or "spaghetti" in nombre:
        return "pasta"
    if "arroz" in nombre or "chaufa" in nombre:
        return "arroz"
    if "salchipapa" in nombre: return "fast_food"
    return "menu"

PLATOS = []
for nom in PLATOS_NOMBRES:
    cat = categoria_plato(nom)
    precio = random.randint(12, 20)       # precio de venta entero
    # costo entre 50% y 70% del precio
    costo = round(precio * random.uniform(0.5, 0.7), 2)
    PLATOS.append((nom, cat, float(precio), costo))

cur.executemany(
    "INSERT INTO platos(nombre,categoria,precio,costo) VALUES (?,?,?,?)",
    PLATOS
)
conn.commit()

# ============================================================
# 5) GENERAR VENTAS (3000 TICKETS)
# ============================================================
MODALIDAD = ["local", "pension"]          # solo estas 2, sin delivery
EXTRAS = ["gaseosa", "postre", "nada"]

START = datetime(2023, 1, 1)
END   = datetime(2024, 12, 31)
days  = (END - START).days

tickets_rows = []
detalle_rows = []

ticket_id = 1

for _ in range(3000):
    fecha = START + timedelta(days=random.randint(0, days))

    id_cliente = random.randint(1, 400)
    modalidad = random.choice(MODALIDAD)

    # cada ticket tiene entre 1 y 4 platos
    n_items = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.10, 0.05])[0]

    platos_elegidos = random.choices(
        range(1, len(PLATOS) + 1),
        k=n_items
    )

    total = 0.0
    for pid in platos_elegidos:
        cantidad = random.choice([1, 1, 1, 2])  # la mayoría pide 1
        precio = PLATOS[pid - 1][2]
        subtotal = precio * cantidad
        extra = random.choice(EXTRAS)
        total += subtotal

        detalle_rows.append((ticket_id, pid, cantidad, subtotal, extra))

    ticket_prom = total / n_items

    tickets_rows.append((
        fecha.strftime("%Y-%m-%d"),
        id_cliente,
        modalidad,
        total,
        ticket_prom
    ))

    ticket_id += 1

cur.executemany(
    "INSERT INTO tickets(fecha,id_cliente,modalidad,total,ticket_promedio) VALUES (?,?,?,?,?)",
    tickets_rows
)

cur.executemany(
    "INSERT INTO detalle_ticket(id_ticket,id_plato,cantidad,subtotal,extra) VALUES (?,?,?,?,?)",
    detalle_rows
)

conn.commit()

# ============================================================
# 6) ROTACIÓN (KPI-3) A PARTIR DE LOS PLATOS MÁS VENDIDOS POR MES
#    Solo guardamos mes, id_plato y cantidad_vendida_mes.
#    La clasificación de rotación la hará el modelo ML.
# ============================================================
rot_dict = {}  # clave: (mes, id_plato) -> cantidad

for row in detalle_rows:
    tid, pid, cant, sub, extra = row

    # tickets_rows[tid-1] = (fecha, id_cliente, modalidad, total, ticket_promedio)
    fecha_ticket = tickets_rows[tid - 1][0]
    mes = fecha_ticket[:7]  # "YYYY-MM"

    clave = (mes, pid)
    rot_dict[clave] = rot_dict.get(clave, 0) + cant

rotacion_rows = []
for (mes, pid), qty in rot_dict.items():
    rotacion_rows.append((mes, pid, qty))

cur.executemany(
    "INSERT INTO rotacion_insumos(mes,id_plato,cantidad_vendida_mes) VALUES (?,?,?)",
    rotacion_rows
)

conn.commit()
conn.close()

print("\nBase de datos realista generada correctamente:")
print(f" → {DB_PATH}")