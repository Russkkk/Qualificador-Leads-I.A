import sqlite3
import pandas as pd

DB_FILE = "leads.db"

df = pd.read_csv("leads_teste.csv")

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Limpa tabela se quiser recomeçar (opcional)
# cursor.execute("DELETE FROM leads")

for _, row in df.iterrows():
cursor.execute("""
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tempo_site INTEGER,
    paginas_visitadas INTEGER,
    clicou_preco INTEGER,
    virou_cliente INTEGER
)
""")

dados = [
    (5, 1, 0, 0),
    (8, 2, 0, 0),
    (12, 3, 1, 0),
    (15, 4, 1, 0),
    (20, 5, 1, 1),
    (25, 6, 1, 1),
    (30, 8, 1, 1)
]

cursor.executemany("""
INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
VALUES (?, ?, ?, ?)
""", dados)

conn.commit()
conn.close()

print("Dados inseridos com sucesso!")

# ==============================================
# =====-------------LEADS-------------==========
# ==============================================

leads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tempo_site INTEGER,
  paginas_visitadas INTEGER,
  clicou_preco INTEGER,
  virou_cliente INTEGER
)

