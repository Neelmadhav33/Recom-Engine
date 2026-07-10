# CineMatch

A simple, content-based movie recommendation system built on the **MovieLens 100K** dataset — no ALS, no collaborative filtering. Genres are turned into TF-IDF vectors and compared with Cosine Similarity to power the recommendations.

## How it works

1. **PySpark** (`preprocessing/preprocess.py`) loads the raw MovieLens files, cleans them, joins ratings + movies + users, and builds a `genre_string` per movie (e.g. `"Animation Children Comedy"`), plus `avg_rating`, `rating_count`, and `popularity_score`.
2. **scikit-learn** (`recommendation/recommend.py`) converts every movie's `genre_string` into a TF-IDF vector, then computes Cosine Similarity between every pair of movies. The result is saved as `models/similarity_matrix.pkl`.
3. **Flask** (`backend/app.py`) serves a small JSON API on top of the similarity matrix, and also serves the frontend.
4. **HTML/CSS/JS** (`frontend/`) is a dark, IMDb-inspired UI: search, popular movies, movie details, recommendations with a `% Match` score + explanation, and a wishlist saved in the browser's `localStorage`.

## Tech stack

- Data processing: PySpark (beginner-friendly DataFrame ops only)
- Recommendation logic: TF-IDF + Cosine Similarity (scikit-learn)
- Backend: Python + Flask
- Frontend: vanilla HTML, CSS, JavaScript

## Project structure

```
project/
├── data/
│   ├── ml-100k/           # raw MovieLens 100K files
│   └── processed/         # cleaned CSVs from PySpark
├── preprocessing/
│   └── preprocess.py      # Stage 2 - PySpark pipeline
├── recommendation/
│   └── recommend.py       # Stage 4 - TF-IDF + Cosine Similarity
├── backend/
│   └── app.py             # Stage 5 - Flask API + static file server
├── frontend/
│   ├── html/index.html
│   ├── css/style.css
│   └── js/app.js
├── models/                 # saved similarity_matrix.pkl + movies_lookup.pkl
└── README.md
```

## Running the project

```bash
# 1. Install dependencies
pip install pyspark scikit-learn pandas flask flask-cors --break-system-packages

# 2. Run PySpark preprocessing (produces data/processed/*.csv)
python3 preprocessing/preprocess.py

# 3. Build the TF-IDF + Cosine Similarity model (produces models/*.pkl)
python3 recommendation/recommend.py

# 4. Start the app
python3 backend/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/search?q=toy` | Search movies by title |
| `GET /api/movie/<movie_id>` | Full details for one movie |
| `GET /api/recommend/<movie_id>` | Top 6 similar movies (genre-based) |
| `GET /api/popular` | Popular movies, ranked by rating + rating count |

## Data used vs. not available

MovieLens 100K provides ratings, titles, release dates, IMDb URLs, and 19 binary genre flags per movie, plus basic user demographics. It does **not** include cast, director, plot/overview, keywords, runtime, budget, revenue, or a poster image — so recommendations here are genre-based, not plot/cast-based.

## Future improvements

- Add a small backend cache for repeated search queries
- Use release year as a secondary weighted signal in the similarity score
- Add basic user-based filtering (e.g. "users like you also liked...") without ALS, using simple co-rating counts
