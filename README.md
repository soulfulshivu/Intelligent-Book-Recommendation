# Intelligent-Book-Recommendation
An intelligent Machine Learning-based book recommendation system using Popularity-Based, Content-Based, Collaborative Filtering, and Hybrid Recommendation techniques to provide personalized book suggestions.
## Project objectives

- Explore and preprocess book, user, and rating data.
- Recommend popular books using rating counts and average ratings.
- Find books with similar metadata using TF-IDF and cosine-distance nearest neighbours.
- Find books with similar user-rating patterns using item-based K-nearest neighbours.
- Combine content similarity and popularity in a simple hybrid score.
- Evaluate a popularity baseline and collaborative item-KNN using a per-user holdout experiment.

## Repository structure

```text
Intelligent-Library-Book-Recommendation/
├── ml-project-intelligent-library-book-recommendation.ipynb
├── archive.zip(datset)
├── Results
├── Requirements
├── .gitignore
└── README.md
```

## Dataset

This notebook is designed for the Book-Crossing dataset, which contains three files:

- `Books.csv` — ISBN, title, author, publication year, publisher, and related metadata.
- `Users.csv` — user identifiers and optional demographic/location fields.
- `Ratings.csv` — user identifier, ISBN, and book rating.

The dataset is included in this repository. 

## Algorithms and methodology

### 1. Popularity-based recommendation
Groups ratings by ISBN, calculates rating count and average rating, and ranks books by
rating count (with average rating available as an additional signal). This is a
non-personalized baseline.

### 2. Content-based recommendation
Combines book title, author, and publisher text. `TfidfVectorizer` converts this text
into TF-IDF features. `NearestNeighbors(metric="cosine")` retrieves books with similar
feature vectors.

### 3. Collaborative filtering
Uses positive user ratings to construct a sparse item-user matrix. K-nearest neighbours
with cosine distance finds books with similar rating patterns.

### 4. Hybrid recommendation
Combines content similarity and a normalized popularity score:

`Hybrid Score = 0.7 × Content Similarity + 0.3 × Popularity Score`

The weights are heuristic project choices, not learned parameters. They can be tuned
with a validation set in future work.

## Experimental evaluation

The notebook includes an offline holdout experiment:

1. Keep users with at least three positive book interactions.
2. Hold out one interaction per selected user.
3. Build the popularity baseline and collaborative item-KNN using training interactions.
4. Use one remaining training book as the seed and check whether the held-out book is
   retrieved in the top 10 recommendations.
5. Report Precision@10, Recall@10, and HitRate@10.

The experiment is a simple retrieval evaluation and has limitations: it uses one seed
book per user, samples at most 1,000 users for runtime, and does not evaluate the
content-based or hybrid models. Run the notebook to generate actual metric values;
do not report unexecuted or assumed numbers. Results are saved to
`/kaggle/working/model_evaluation_results.csv`.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The notebook uses Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn.
Kaggle Notebooks already provide most of these packages.

## How to run on Kaggle

1. Sign in to Kaggle and open **Code → New Notebook**.
2. Use **Add Input** to attach the Book-Crossing dataset.
3. Open the notebook in `notebooks/` or upload it to Kaggle.
4. Run the dataset-discovery cell to inspect the actual file paths.
5. Update `dataset_path` in the data-loading cell so it points to the directory
   containing `Books.csv`, `Users.csv`, and `Ratings.csv`.
6. Select **Run All**.
7. Inspect recommendation outputs, visualizations, and the evaluation table.
8. Download generated CSV files from `/kaggle/working/`.

If your dataset has different file names or column names, adjust the loading and
preprocessing cells accordingly.

## Notes and limitations

- Book titles, authors, and publishers are used as text features; full descriptions
  and genre labels are not used in this version.
- The collaborative model is based on explicit positive ratings and does not use
  zero-valued Book-Crossing ratings as positive feedback.
- The hybrid weighting is heuristic.
- Similarity is not the same as recommendation accuracy.
- Cold-start users and books with few ratings remain challenging.
- Dataset files are excluded from Git because of their size and licensing/source terms.

## Future improvements

- Add genre and book-description features.
- Tune hybrid weights using validation data.
- Evaluate all recommenders with consistent ranking metrics.
- Add a user-facing interface using Streamlit.
- Add user profiles, filters, and explainable recommendation reasons.
