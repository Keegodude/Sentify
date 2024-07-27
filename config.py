# config.py
import secrets

# Spotify and genius API credentials
FLASK_SECRET_KEY = secrets.token_hex(32)
SPOTIPY_CLIENT_ID = '7068594a5bae4571b8e2c8e2830a84a1'
SPOTIPY_CLIENT_SECRET = '8033348c6630455d94cc297bdf4dbfc3'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'
GENIUS_API_TOKEN = '5I7dm5CNtiuKA6rVhHJjDM7WJfZdoKIpH3r4U5Gq6qOoi77kl1VPCFmGjWax3fdq'
