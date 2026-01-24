import sqlite3
import pandas as pd

DB_FILE = "leads.db"

df = pd.read_csv("leads_teste.csv")

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Limpa tabela se quiser recomeçar (opcional)
# cursor.execute("DELETE FROM leads")

cursor.execute("""
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tempo_site INTEGER,
    paginas_visitadas INTEGER,
    clicou_preco INTEGER,
    virou_cliente INTEGER
)
""")

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO leads (tempo_site, paginas_visitadas, clicou_preco, virou_cliente)
        VALUES (?, ?, ?, ?)
    """, (row['tempo_site'], row['paginas_visitadas'], row['clicou_preco'], row['virou_cliente']))

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
