import sqlite3

DB_FILE = "leads.db"

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Criar tabela corretamente
cursor.execute("""
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tempo_site INTEGER,
    paginas_visitadas INTEGER,
    clicou_preco INTEGER,
    virou_cliente INTEGER
)
""")

# Dados realistas (frius + quentes)
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

print("✅ Banco populado com dados de treino!")
