# Movie Data Analysis with Python

This Python project explores movie performance using metadata, ratings, genres, revenue, budget, profit, and ROI. The project was completed in Jupyter Notebook and converted into a reusable Python script for the repository.

## Business Objective

The analysis looks at which movie attributes are associated with stronger audience response and commercial performance.

## Tools

- Python
- pandas
- Jupyter Notebook
- Matplotlib
- Seaborn

## Data Source

The analysis uses a public movies dataset from Kaggle, including movie metadata, ratings, links, credits, and keywords.

## Repository Structure

```text
notebooks/
  movie_data_analysis_case_study.ipynb

scripts/
  movie_analysis.py

docs/
  analysis_workflow.md

requirements.txt
README.md
```

## Project Workflow

The dataset was loaded and inspected in Jupyter Notebook. The work included missing-value checks, duplicate review, data type corrections, financial metric creation, genre extraction, and visual analysis. The notebook presents the exploratory process, while the Python script keeps the analysis logic reusable.

## Key Insights

- Drama appeared most frequently in the dataset.
- Family and Animation showed strong average financial performance among movies with usable budget and revenue data.
- Budget and revenue fields required careful filtering because many records were incomplete.
- Audience ratings and financial performance measure different dimensions of success.

## Recommendations

- Analyze audience quality and financial performance separately.
- Use genre, popularity, budget, and release timing together when evaluating movie success.
- Treat missing financial data as a core limitation when interpreting revenue and ROI trends.
