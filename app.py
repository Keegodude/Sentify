import numpy as np
import unicodedata
from bokeh.io import output_file, save
from flask import Flask, request, redirect, session, url_for, render_template, send_file
from flask_session import Session
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
import pandas as pd
from textblob import TextBlob
import config
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import re
from flask_sqlalchemy import SQLAlchemy
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Flask App initialization
app = Flask(__name__)

app.secret_key = config.FLASK_SECRET_KEY
app.config['SESSION_COOKIE_NAME'] = 'your_session_cookie'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Define spotify/genius credentials
sp_oauth = SpotifyOAuth(client_id=config.SPOTIPY_CLIENT_ID,
                        client_secret=config.SPOTIPY_CLIENT_SECRET,
                        redirect_uri=config.SPOTIPY_REDIRECT_URI,
                        scope='playlist-read-private')
genius = lyricsgenius.Genius(config.GENIUS_API_TOKEN)

# Database initialization
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///playlists.db'
db = SQLAlchemy(app)


class Playlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    playlist_id = db.Column(db.String(100), unique=True, nullable=False)
    playlist_name = db.Column(db.String(100), nullable=False)
    avg_polence = db.Column(db.Float, nullable=False)
    owner_id = db.Column(db.Integer, unique=False)
    owner_username = db.Column(db.String(100), nullable=False)

    def __init__(self, playlist_id, playlist_name, avg_polence, owner_id, owner_username):
        self.playlist_id = playlist_id
        self.playlist_name = playlist_name
        self.avg_polence = avg_polence
        self.owner_id = owner_id
        self.owner_username = owner_username


class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    song_id = db.Column(db.String(100), unique=True, nullable=False)
    song_name = db.Column(db.String(255), nullable=False)
    artist = db.Column(db.String(255), nullable=False)
    lyrics = db.Column(db.Text, nullable=True)
    valence = db.Column(db.Float, nullable=False)
    polarity = db.Column(db.Float, nullable=True)
    polence = db.Column(db.Float, nullable=True)
    word_sentiments = db.Column(db.JSON, nullable=True)
    danceability = db.Column(db.Float, nullable=False)
    energy = db.Column(db.Float, nullable=False)
    tempo = db.Column(db.Float, nullable=False)
    acousticness = db.Column(db.Float, nullable=False)
    instrumentalness = db.Column(db.Float, nullable=False)
    loudness = db.Column(db.Float, nullable=False)
    speechiness = db.Column(db.Float, nullable=False)

    def __init__(self, song_id, song_name, artist, lyrics, valence, polarity, polence, word_sentiments, danceability,
                 energy, tempo,
                 acousticness, instrumentalness, loudness, speechiness):
        self.song_id = song_id
        self.song_name = song_name
        self.artist = artist
        self.lyrics = lyrics
        self.valence = valence
        self.polarity = polarity
        self.polence = polence
        self.word_sentiments = word_sentiments
        self.danceability = danceability
        self.energy = energy
        self.tempo = tempo
        self.acousticness = acousticness
        self.instrumentalness = instrumentalness
        self.loudness = loudness
        self.speechiness = speechiness


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)


@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('playlists'))


def get_token():
    token_info = session.get('token_info', None)
    if not token_info:
        return None
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    return token_info


@app.route('/playlists')
def playlists():
    token_info = get_token()
    sp = spotipy.Spotify(auth=token_info['access_token'])
    playlists = sp.current_user_playlists()
    return render_template('playlists.html', playlists=playlists['items'])


def generate_distribution_plot(df, feature):
    plt.figure(figsize=(10, 6))
    plot = sns.histplot(df[feature], kde=True)
    buf = io.BytesIO()
    plot.figure.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def plot_polence_distribution(current_polence, all_polences):
    plt.figure(figsize=(10, 6))
    plot = sns.histplot(all_polences, bins=20, kde=True, color='blue', label='All Playlists')
    plt.axvline(current_polence, color='red', linestyle='dashed', linewidth=2, label='Current Playlist')
    plt.legend()
    plt.title('Distribution of Average Polence')
    plt.xlabel('Polence')
    plt.ylabel('Frequency')
    buf = io.BytesIO()
    plot.figure.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def generate_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    buf = io.BytesIO()
    heatmap.figure.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def generate_pca_plot(df, optimal_k=3):
    features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness',
                'instrumentalness', 'loudness', 'speechiness', 'polarity', 'polence']

    # Drop rows with missing values in the features
    df_clean = df.dropna(subset=features)
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    # Fit K-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    df_clean['cluster'] = kmeans.labels_

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # Create a DataFrame with the PCA results
    df_clean['pc1'] = X_pca[:, 0]
    df_clean['pc2'] = X_pca[:, 1]

    # Create PCA plot
    plt.figure(figsize=(12, 8))
    pca_plot = sns.scatterplot(x='pc1', y='pc2', hue='cluster', data=df_clean, palette='viridis')
    # Label every other datapoint for clarity
    for i, row in df_clean.iloc[::2].iterrows():
        plt.text(row['pc1'] + .02, row['pc2'], row['name'], fontsize=6)

    plt.title('Cluster Visualization using PCA')
    plt.xlabel(f'pc1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
    plt.ylabel(f'pc2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    pca_plot.figure.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def generate_bokeh_pca(df, optimal_k=3):
    # Assuming df is your DataFrame and optimal_k is the number of clusters
    features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness',
                'instrumentalness', 'loudness', 'speechiness', 'polarity', 'polence']

    # Drop rows with missing values in the features
    df_clean = df.dropna(subset=features)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])

    # Fit K-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    df_clean['cluster'] = kmeans.labels_

    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with the PCA results
    df_clean['pc1'] = X_pca[:, 0]
    df_clean['pc2'] = X_pca[:, 1]

    # Create a new plot
    p = figure(title='Cluster Visualization using PCA')

    # Add a circle renderer with vectorized colors and sizes
    source = ColumnDataSource(data=dict(
        x=df_clean['pc1'],
        y=df_clean['pc2'],
        color=df_clean['cluster'],
        song_name=df_clean['name'],
    ))

    p.circle('x', 'y', color='color', source=source)

    # Add a hover tool referring to the formatted columns
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(PCA1,PCA2)", "(@x, @y)"),
        ("song", "@song_name"),
    ])

    p.add_tools(hover)
    output_file("templates/pca_plot.html")
    # Show the result
    save(p)


@app.route('/analyze/<playlist_id>')
def analyze(playlist_id):
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    sp = spotipy.Spotify(auth=token_info['access_token'])
    playlist_name = sp.playlist(playlist_id)['name']
    playlist_owner = sp.playlist(playlist_id)['owner']
    owner_id = playlist_owner['id']
    owner_username = playlist_owner['display_name']
    df_tracks = get_playlist_tracks_audio_features(sp, playlist_id)

    df_tracks.sort_values(by='polence', ascending=False, inplace=True)

    # Generate correlation heatmap
    correlation_heatmap = generate_correlation_heatmap(df_tracks)

    # Generate distribution plot for valence
    valence_distribution = generate_distribution_plot(df_tracks, 'valence')

    # Generate distribution plot for polarity
    polarity_distribution = generate_distribution_plot(df_tracks, 'polarity')

    # Generate dist plot for polence
    polence_distribution = generate_distribution_plot(df_tracks, 'polence')

    # Generate PCA-based k-means cluster plot
    pca_plot = generate_pca_plot(df_tracks, )

    # Get avg polarity of playlist
    avg_polence = df_tracks['polence'].mean()

    # Plot playlist's avg polence against all others
    all_polences = [playlist.avg_polence for playlist in Playlist.query.all()]
    polences_mean = np.mean(all_polences)
    polences_median = np.median(all_polences)
    current_polence = avg_polence
    interpolence_distribution = plot_polence_distribution(current_polence, all_polences)

    # db handling
    existing_playlist = Playlist.query.filter_by(playlist_id=playlist_id).first()
    if existing_playlist:
        # Update the existing entry
        existing_playlist.avg_polence = avg_polence
    else:
        # Create a new entry
        new_playlist = Playlist(playlist_id, playlist_name, avg_polence, owner_id, owner_username)
        db.session.add(new_playlist)

    db.session.commit()
    # Generate descriptive statistics
    descriptive_stats = df_tracks.describe()

    if 'analysis_result' not in session:
        session['analysis_result'] = {}
    session['analysis_result'][playlist_id] = df_tracks.to_dict('records')

    df_tracks.drop(columns=['lyrics'], inplace=True)

    return render_template('analysis.html',
                           tables=[df_tracks.to_html(classes='data'),
                                   descriptive_stats.to_html(classes='data')],
                           titles=['Tracks', 'Correlation Matrix', 'Descriptive Statistics'],
                           playlist_name=playlist_name,
                           heatmap=correlation_heatmap,
                           playlist_avg_polence=avg_polence,
                           valence_distribution=valence_distribution,
                           polarity_distribution=polarity_distribution,
                           polence_distribution=polence_distribution,
                           pca_plot=pca_plot,
                           interpolence_distribution=interpolence_distribution,
                           polences_mean=polences_mean,
                           polences_median=polences_median,
                           playlist_id=playlist_id)


@app.route('/song_info/<playlist_id>')
def song_info(playlist_id):
    # Retrieve the song info from the session or database
    if 'analysis_result' in session and playlist_id in session['analysis_result']:
        df_tracks = pd.DataFrame(session['analysis_result'][playlist_id])
    else:
        # Fallback to fetching from database or re-analysis if needed
        token_info = get_token()
        if not token_info:
            return redirect(url_for('login'))
        sp = spotipy.Spotify(auth=token_info['access_token'])
        df_tracks = get_playlist_tracks_audio_features(sp, playlist_id)

    return render_template('song_info.html', tables=[df_tracks.to_html(classes='data')], titles=['Tracks'])


@app.route('/download_csv/<playlist_id>')
def download_csv(playlist_id):
    if 'analysis_result' in session and session['analysis_result'].get(playlist_id):
        df_tracks = pd.DataFrame(session['analysis_result'][playlist_id])
        # Create a buffer to hold the CSV data
        buf = io.BytesIO()
        df_tracks.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(buf, mimetype='text/csv', download_name=f'playlist_{playlist_id}.csv', as_attachment=True)
    return 'No analysis data found for the specified playlist.'


def get_playlist_tracks_audio_features(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])

    track_data = []
    for item in tracks:
        track = item['track']
        if track is None:
            continue
        track_id = track['id']
        if track_id is None:
            continue
        song = Song.query.filter_by(song_id=track_id).first()
        if song:
            song_name = song.song_name
            artist = song.artist
            lyrics = song.lyrics
            sentiment, word_sentiments = analyze_sentiment(lyrics) if lyrics else (None, [])
            valence = song.valence
            polarity = song.polarity
            polence = song.polence
            five_word_sentiments = song.word_sentiments
            danceability = song.danceability
            energy = song.energy
            tempo = song.tempo
            acousticness = song.acousticness
            instrumentalness = song.instrumentalness
            loudness = song.loudness
            speechiness = song.speechiness
        else:
            audio_features = sp.audio_features(track['id'])[0]
            song_name = track['name']
            artist = track['artists'][0]['name']
            if contains_east_asian_chars(song_name) or contains_east_asian_chars(artist):
                lyrics = None
            else:
                lyrics = get_lyrics(song_name, artist)
            sentiment, word_sentiments = analyze_sentiment(lyrics) if lyrics else (None, [])
            valence = float(audio_features['valence'])
            polarity = float(sentiment.polarity) if sentiment else None
            normalized_polarity = float((sentiment.polarity + 1) / 2) if sentiment else None
            polence = float(((.8 * valence) + (.2 * normalized_polarity))) if normalized_polarity else None
            five_word_sentiments = word_sentiments[:5] + word_sentiments[-5:] if word_sentiments else []
            danceability = float(audio_features['danceability'])
            energy = float(audio_features['energy'])
            tempo = float(audio_features['tempo'])
            acousticness = float(audio_features['acousticness'])
            instrumentalness = float(audio_features['instrumentalness'])
            loudness = float(audio_features['loudness'])
            speechiness = float(audio_features['speechiness'])

            new_song = Song(track_id, song_name, artist, lyrics, valence, polarity, polence, five_word_sentiments,
                            danceability, energy,
                            tempo, acousticness, instrumentalness, loudness, speechiness)
            db.session.add(new_song)
            db.session.commit()

        # append song info to track_data
        track_info = {
            'name': song_name,
            'artist': artist,
            'lyrics': lyrics,
            'polarity': polarity if sentiment else None,
            'valence': valence,
            'polence': polence,
            'word sentiments': five_word_sentiments,
            'danceability': danceability,
            'energy': energy,
            'tempo': tempo,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'loudness': loudness,
            'speechiness': speechiness
        }
        track_data.append(track_info)
    return pd.DataFrame(track_data)


def not_english(text):
    # Define a regex pattern for English characters and common symbols on a QWERTY keyboard
    pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]')
    return bool(pattern.match(text))


def contains_east_asian_chars(s):
    for c in s:
        name = unicodedata.name(c)
        if "CJK UNIFIED" in name \
                or "HIRAGANA" in name \
                or "KATAKANA" in name \
                or "HANGUL" in name:
            return True
    return False


def get_lyrics(song_title, artist_name):
    try:
        song = genius.search_song(song_title, artist_name)
        if song:
            lyrics = song.lyrics
            # Split the lyrics into lines, skip the first line, and join them back together
            lyrics_lines = lyrics.split('\n')
            cleaned_lyrics = '\n'.join(lyrics_lines[1:])
            # Remove all text within square brackets
            cleaned_lyrics = re.sub(r'\[.*?]', '', cleaned_lyrics)
            # Remove any extra whitespace
            cleaned_lyrics = re.sub(r'\s+', ' ', cleaned_lyrics).strip()
            # Some lyrics are bugged and seem to be mis-targeted,
            # for now this fix avoids that by nullifying entries that are too long
            dash_count = cleaned_lyrics.count('-')
            cap_count = len(re.findall(r'[A-Z]', cleaned_lyrics))
            if cap_count > 200 and dash_count > 40 or not_english(cleaned_lyrics):
                cleaned_lyrics = None
            return cleaned_lyrics
        else:
            return None
    except Exception as e:
        print(f"Error retrieving lyrics for {song_title} by {artist_name}: {e}")
        return None


def analyze_sentiment(lyrics):
    analysis = TextBlob(lyrics)
    words = analysis.words
    word_sentiments = [(word, TextBlob(word).sentiment.polarity) for word in words]
    non_zero_word_sentiments = [ws for ws in word_sentiments if ws[1] != 0]
    sorted_sentiments = sorted(non_zero_word_sentiments, key=lambda x: x[1], reverse=True)
    return analysis.sentiment, sorted_sentiments


if __name__ == '__main__':
    app.run(debug=True)
