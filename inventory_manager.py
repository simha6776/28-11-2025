# inventory_manager.py
# modules/inventory_manager.py
import sqlite3
from datetime import datetime

DB_NAME = "inventory.db"

class InventoryManager:
    def __init__(self):
        self.conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            quantity INTEGER,
            price REAL,
            category TEXT,
            last_update TEXT,
            sold INTEGER DEFAULT 0
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def add_item(self, name, quantity, price, category):
        query = """INSERT INTO inventory (name, quantity, price, category, last_update)
                   VALUES (?, ?, ?, ?, ?)"""
        self.conn.execute(query, (name, quantity, price, category, datetime.now()))
        self.conn.commit()

    def update_item(self, item_id, name, quantity, price, category):
        query = """UPDATE inventory
                   SET name=?, quantity=?, price=?, category=?, last_update=?
                   WHERE id=?"""
        self.conn.execute(query, (name, quantity, price, category, datetime.now(), item_id))
        self.conn.commit()

    def delete_item(self, item_id):
        query = "DELETE FROM inventory WHERE id=?"
        self.conn.execute(query, (item_id,))
        self.conn.commit()

    def get_all_items(self):
        return self.conn.execute("SELECT * FROM inventory").fetchall()

    def search_item(self, keyword):
        keyword = f"%{keyword}%"
        query = "SELECT * FROM inventory WHERE name LIKE ? OR category LIKE ?"
        return self.conn.execute(query, (keyword, keyword)).fetchall()

    def reset_inventory(self):
        self.conn.execute("DELETE FROM inventory")
        self.conn.commit()

    def record_sale(self, item_id, qty_sold):
        row = self.conn.execute(
            "SELECT quantity, sold FROM inventory WHERE id=?", (item_id,)
        ).fetchone()

        if not row:
            return False

        qty_left = row[0] - qty_sold
        if qty_left < 0:
            return False

        new_sold = row[1] + qty_sold

        self.conn.execute(
            """UPDATE inventory 
               SET quantity=?, sold=?, last_update=?
               WHERE id=?""",
            (qty_left, new_sold, datetime.now(), item_id),
        )
        self.conn.commit()
        return True

    def low_stock_alerts(self, threshold=5):
        return self.conn.execute(
            "SELECT * FROM inventory WHERE quantity <= ?", (threshold,)
        ).fetchall()
