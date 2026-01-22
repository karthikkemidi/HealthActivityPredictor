import jwt
import bcrypt
import datetime
from functools import wraps
import streamlit as st
SECRET_KEY = st.secrets.get("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"

class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    @staticmethod
    def create_token(user_id, participant_id, username: str) -> str:
        import time
        
        # Handle bytes conversion from database
        if isinstance(user_id, bytes):
            user_id = int.from_bytes(user_id, byteorder='little')
        if isinstance(participant_id, bytes):
            participant_id = int.from_bytes(participant_id, byteorder='little')
        
        now = datetime.datetime.utcnow()
        exp_time = now + datetime.timedelta(hours=24)
        
        payload = {
            'user_id': int(user_id),
            'participant_id': int(participant_id),
            'username': str(username),
            'exp': int(exp_time.timestamp()),
            'iat': int(now.timestamp())
        }
        
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        
        if isinstance(token, bytes):
            return token.decode('utf-8')
        return str(token)

    
    @staticmethod
    def decode_token(token: str) -> dict:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def get_current_user():
        if 'auth_token' in st.session_state:
            payload = AuthManager.decode_token(st.session_state['auth_token'])
            if payload:
                return payload
        return None
    
    @staticmethod
    def is_authenticated() -> bool:
        return AuthManager.get_current_user() is not None
    
    @staticmethod
    def logout():
        if 'auth_token' in st.session_state:
            del st.session_state['auth_token']
        if 'user' in st.session_state:
            del st.session_state['user']

def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not AuthManager.is_authenticated():
            st.warning("⚠️ Please login to access this page")
            st.stop()
        return func(*args, **kwargs)
    return wrapper
