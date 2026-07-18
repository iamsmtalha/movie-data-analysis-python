"""Python movie data analysis project.

This project follows the Google Data Analytics case study process:
Ask, Prepare, Process, Analyze, Share, Act.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data_raw"
CLEAN_DIR = PROJECT_DIR / "data_clean"
VISUALS_DIR = PROJECT_DIR / "visuals"

MOVIES_FILE = RAW_DIR / "movies_metadata.csv"
RATINGS_FILE = RAW_DIR / "ratings_small.csv"
LINKS_FILE = RAW_DIR / "links_small.csv"
CREDITS_FILE = RAW_DIR / "credits.csv"
KEYWORDS_FILE = RAW_DIR / "keywords.csv"

CLEAN_MOVIES_FILE = CLEAN_DIR / "movies_analysis_cleaned.csv"
RATING_SUMMARY_FILE = CLEAN_DIR / "movie_rating_summary.csv"


def parse_list_field(value: object) -> list[dict]:
    """Parse TMDb list-like text fields such as genres, cast, crew, and keywords."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def get_names(value: object) -> list[str]:
    return [item.get("name") for item in parse_list_field(value) if isinstance(item, dict) and item.get("name")]


def get_primary_genre(value: object) -> str | None:
    names = get_names(value)
    return names[0] if names else None


def get_director(crew_value: object) -> str | None:
    for item in parse_list_field(crew_value):
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name")
    return None


def get_top_cast(cast_value: object, limit: int = 3) -> str:
    names = get_names(cast_value)
    return ", ".join(names[:limit])


def load_movies() -> pd.DataFrame:
    movies = pd.read_csv(MOVIES_FILE, low_memory=False)

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id"]).copy()
    movies["id"] = movies["id"].astype("int64")

    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["year"] = movies["release_date"].dt.year

    numeric_columns = ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]
    for column in numeric_columns:
        movies[column] = pd.to_numeric(movies[column], errors="coerce")

    movies["primary_genre"] = movies["genres"].apply(get_primary_genre)
    movies["genre_list"] = movies["genres"].apply(lambda value: ", ".join(get_names(value)))
    movies["profit"] = movies["revenue"] - movies["budget"]
    movies["roi"] = movies["profit"] / movies["budget"]
    movies.loc[movies["budget"].le(0), "roi"] = pd.NA
    movies["has_financial_data"] = movies["budget"].gt(0) & movies["revenue"].gt(0)

    movies = movies[
        [
            "id",
            "title",
            "original_title",
            "release_date",
            "year",
            "primary_genre",
            "genre_list",
            "original_language",
            "budget",
            "revenue",
            "profit",
            "roi",
            "runtime",
            "popularity",
            "vote_average",
            "vote_count",
            "status",
            "has_financial_data",
        ]
    ]

    movies = movies.drop_duplicates(subset=["id"])
    movies = movies[movies["title"].notna()].copy()
    return movies


def load_ratings_summary() -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS_FILE)
    links = pd.read_csv(LINKS_FILE)

    rating_summary = (
        ratings.groupby("movieId")
        .agg(
            avg_user_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
        )
        .reset_index()
    )

    rating_summary = rating_summary.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    rating_summary["tmdbId"] = pd.to_numeric(rating_summary["tmdbId"], errors="coerce")
    rating_summary = rating_summary.dropna(subset=["tmdbId"]).copy()
    rating_summary["tmdbId"] = rating_summary["tmdbId"].astype("int64")
    return rating_summary


def add_credits(movies: pd.DataFrame) -> pd.DataFrame:
    credits = pd.read_csv(CREDITS_FILE)
    credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
    credits = credits.dropna(subset=["id"]).copy()
    credits["id"] = credits["id"].astype("int64")
    credits = credits.drop_duplicates(subset=["id"])
    credits["director"] = credits["crew"].apply(get_director)
    credits["top_cast"] = credits["cast"].apply(get_top_cast)
    return movies.merge(credits[["id", "director", "top_cast"]], on="id", how="left")


def add_keywords(movies: pd.DataFrame) -> pd.DataFrame:
    keywords = pd.read_csv(KEYWORDS_FILE)
    keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")
    keywords = keywords.dropna(subset=["id"]).copy()
    keywords["id"] = keywords["id"].astype("int64")
    keywords = keywords.drop_duplicates(subset=["id"])
    keywords["keywords_text"] = keywords["keywords"].apply(lambda value: ", ".join(get_names(value)))
    return movies.merge(keywords[["id", "keywords_text"]], on="id", how="left")


def build_clean_dataset() -> pd.DataFrame:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    movies = load_movies()
    rating_summary = load_ratings_summary()
    movies = movies.merge(
        rating_summary,
        left_on="id",
        right_on="tmdbId",
        how="left",
    ).drop(columns=["tmdbId"])
    movies = add_credits(movies)
    movies = add_keywords(movies)

    movies.to_csv(CLEAN_MOVIES_FILE, index=False)
    rating_summary.to_csv(RATING_SUMMARY_FILE, index=False)
    return movies


def analyze(movies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    financial = movies[movies["has_financial_data"]].copy()

    genre_counts = (
        movies.groupby("primary_genre", dropna=False)
        .size()
        .reset_index(name="movie_count")
        .sort_values("movie_count", ascending=False)
    )

    genre_financials = (
        financial.groupby("primary_genre", dropna=False)
        .agg(
            movie_count=("title", "count"),
            avg_budget=("budget", "mean"),
            avg_revenue=("revenue", "mean"),
            avg_profit=("profit", "mean"),
            avg_roi=("roi", "mean"),
            avg_vote=("vote_average", "mean"),
        )
        .reset_index()
        .sort_values("avg_profit", ascending=False)
    )

    yearly_trend = (
        movies.dropna(subset=["year"])
        .groupby("year")
        .agg(
            movie_count=("title", "count"),
            avg_vote=("vote_average", "mean"),
            avg_runtime=("runtime", "mean"),
        )
        .reset_index()
        .sort_values("year")
    )

    top_profit = financial.sort_values("profit", ascending=False).head(15)
    top_roi = financial[financial["budget"].ge(100_000)].sort_values("roi", ascending=False).head(15)

    rating_comparison = movies[movies["user_rating_count"].ge(30)].copy()
    rating_comparison = rating_comparison.sort_values("avg_user_rating", ascending=False).head(20)

    return {
        "genre_counts": genre_counts,
        "genre_financials": genre_financials,
        "yearly_trend": yearly_trend,
        "top_profit": top_profit,
        "top_roi": top_roi,
        "rating_comparison": rating_comparison,
    }


def save_outputs(results: dict[str, pd.DataFrame]) -> None:
    for name, table in results.items():
        table.to_csv(CLEAN_DIR / f"{name}.csv", index=False)


def save_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=data, x=x, y=y, color="#2563eb")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / filename, dpi=160)
    plt.close()


def save_horizontal_bar_chart(data: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=data, x=x, y=y, color="#2563eb")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / filename, dpi=160)
    plt.close()


def create_visuals(movies: pd.DataFrame, results: dict[str, pd.DataFrame]) -> None:
    save_bar_chart(
        results["genre_counts"].head(12),
        "primary_genre",
        "movie_count",
        "Top Movie Genres by Count",
        "top_movie_genres.png",
    )

    financials = results["genre_financials"].head(10).copy()
    financials["avg_revenue_millions"] = financials["avg_revenue"] / 1_000_000
    save_bar_chart(
        financials,
        "primary_genre",
        "avg_revenue_millions",
        "Average Revenue by Genre",
        "avg_revenue_by_genre.png",
    )

    profit_genres = results["genre_financials"].head(10).copy()
    profit_genres["avg_profit_millions"] = profit_genres["avg_profit"] / 1_000_000
    save_bar_chart(
        profit_genres,
        "primary_genre",
        "avg_profit_millions",
        "Average Profit by Genre",
        "avg_profit_by_genre.png",
    )

    plt.figure(figsize=(10, 6))
    financial = movies[movies["has_financial_data"]]
    sns.scatterplot(data=financial, x="budget", y="revenue", alpha=0.35)
    plt.title("Budget vs Revenue")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "budget_vs_revenue.png", dpi=160)
    plt.close()

    top_profit = results["top_profit"].head(10).copy()
    top_profit["profit_billions"] = top_profit["profit"] / 1_000_000_000
    save_horizontal_bar_chart(
        top_profit.sort_values("profit_billions", ascending=True),
        "profit_billions",
        "title",
        "Top Movies by Profit",
        "top_movies_by_profit.png",
    )

    top_rated = results["rating_comparison"].head(10).copy()
    save_horizontal_bar_chart(
        top_rated.sort_values("avg_user_rating", ascending=True),
        "avg_user_rating",
        "title",
        "Top User-Rated Movies",
        "top_user_rated_movies.png",
    )

    modern_years = results["yearly_trend"][results["yearly_trend"]["year"].ge(1980)].copy()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=modern_years, x="year", y="movie_count", color="#2563eb")
    plt.title("Movie Releases by Year Since 1980")
    plt.xlabel("Release Year")
    plt.ylabel("Movie Count")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "movie_releases_by_year.png", dpi=160)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    movies = build_clean_dataset()
    results = analyze(movies)
    save_outputs(results)
    create_visuals(movies, results)

    print("Cleaned movie rows:", len(movies))
    print("Cleaned dataset:", CLEAN_MOVIES_FILE)
    print("Summary tables:", CLEAN_DIR)
    print("Visuals:", VISUALS_DIR)
    print("\nTop genres:")
    print(results["genre_counts"].head(10).to_string(index=False))
    print("\nTop profit movies:")
    print(results["top_profit"][["title", "primary_genre", "year", "profit"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
