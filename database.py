import sqlite3

def init_db():
    # This creates a database file named 'newsverify.db' in your folder
    conn = sqlite3.connect('newsverify.db')
    cursor = conn.cursor()

    # Create the main verifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT NOT NULL,
            original_input TEXT NOT NULL,
            extracted_claim TEXT,
            assessment TEXT,
            confidence INTEGER,
            summary TEXT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create the sources table for web evidence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verification_id INTEGER,
            url TEXT NOT NULL,
            title TEXT,
            domain TEXT,
            snippet TEXT,
            stance TEXT,
            published_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (verification_id) REFERENCES verifications (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print("Database and tables initialized successfully!")

if __name__ == '__main__':
    init_db()