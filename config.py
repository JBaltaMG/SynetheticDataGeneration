import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AUTHORITY = os.getenv("AUTHORITY")
SCOPE = os.getenv("SCOPE")
REDIRECT_URI = os.getenv("REDIRECT_URI")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
UPLOADS_FOLDER = os.getenv("UPLOADS_FOLDER")
TENANT_ID = os.getenv("TENANT_ID")
SECRET_SESSION_KEY = os.getenv("SECRET_SESSION_KEY")
if not CLIENT_ID or not CLIENT_SECRET or not AUTHORITY or not SCOPE or not REDIRECT_URI:
    raise ValueError("Missing required environment variables for Azure authentication.")