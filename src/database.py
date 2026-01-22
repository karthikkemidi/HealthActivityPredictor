import sqlite3
import pandas as pd
from typing import Optional, Tuple
from src.auth import AuthManager

def _decode_if_bytes(value):
    """Convert bytes to appropriate type"""
    if isinstance(value, bytes):
        try:
            return int.from_bytes(value, byteorder='little')
        except:
            return value.decode('utf-8')
    return value

class UserDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Force text mode for all columns
        conn.text_factory = str
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                participant_id INTEGER UNIQUE NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_default_users(self, df: pd.DataFrame):
        conn = self.get_connection()
        cursor = conn.cursor()
        participants = df['participant_id'].unique()
        created_count = 0
        
        for pid in participants[:100]:
            username = f"user{pid}"
            email = f"user{pid}@healthapp.com"
            password = f"health{pid}"
            
            try:
                password_hash = AuthManager.hash_password(password)
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, participant_id, full_name)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, email, password_hash, int(pid), f"Participant {pid}"))
                created_count += 1
            except:
                continue
        
        conn.commit()
        conn.close()
        return created_count
    
    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, password_hash, participant_id, email, full_name
            FROM users WHERE username = ? OR email = ?
        ''', (username, username))
        
        result = cursor.fetchone()
        
        if result:
            user_id, username, password_hash, participant_id, email, full_name = result
            
            # Ensure proper types
            user_id = int(user_id) if user_id else None
            participant_id = int(participant_id) if participant_id else None
            
            if user_id and AuthManager.verify_password(password, str(password_hash)):
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?', (user_id,))
                conn.commit()
                conn.close()
                
                return {
                    'user_id': user_id,
                    'username': str(username),
                    'participant_id': participant_id,
                    'email': str(email),
                    'full_name': str(full_name) if full_name else None
                }
        
        conn.close()
        return None
    
    def register_user(self, username: str, email: str, password: str, participant_id: int, full_name: str = None):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = AuthManager.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, participant_id, full_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, int(participant_id), full_name))
            conn.commit()
            conn.close()
            return True, "User registered successfully!"
        except Exception as e:
            conn.close()
            return False, str(e)
    
    def get_user_by_id(self, user_id: int):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, participant_id, email, full_name, created_at, last_login
            FROM users WHERE user_id = ?
        ''', (int(user_id),))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'user_id': int(result[0]),
                'username': str(result[1]),
                'participant_id': int(result[2]),
                'email': str(result[3]),
                'full_name': str(result[4]) if result[4] else None,
                'created_at': str(result[5]),
                'last_login': str(result[6]) if result[6] else None
            }
        return None
    
    def update_password(self, user_id: int, new_password: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        password_hash = AuthManager.hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE user_id = ?', 
                      (password_hash, int(user_id)))
        conn.commit()
        conn.close()
        return True
