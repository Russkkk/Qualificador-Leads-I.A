import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT UNIQUE,
    email TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()
conn.close()

print("Banco de usuários criado com sucesso")