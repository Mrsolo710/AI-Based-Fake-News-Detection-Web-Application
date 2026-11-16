import sqlite3

# Connect to the database
conn = sqlite3.connect('newsverify.db')
cursor = conn.cursor()

# Fetch all verifications
cursor.execute("SELECT id, extracted_claim, assessment FROM verifications")
rows = cursor.fetchall()

print("--- SAVED VERIFICATIONS ---")
for row in rows:
    print(f"ID: {row[0]} | Claim: '{row[1]}' | AI Assessment: {row[2]}")

conn.close()