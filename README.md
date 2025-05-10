## Recommender Systems 

An exploration and review of *Non-Personalised* and *Collaborative-Filtering based* recommender methods (including *Matrix Factorization* for the Recommender Systems course taught by **Marc Torrens, PhD**
at ESADE Ramón Llull University. 

#### Data Used
The dataset used to run and test all the implemented systems was the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/). All the corresponding csv files can be found in `data`. 

#### Content and Structure
- [systems/utils](github.com/roodriigoooo/recsysreview/tree/main/systems/utils) contains auxiliary similarity computation methods: Pearson, Constrained Pearson, Cosine, Jaccard, Euclidean, and Manhattan.
- [systems/nonpersonalised.ipynb](github.com/roodriigoooo/recsysreview/tree/main/systems/nonpersonalised.ipynb) includes initial data exploration, preprocessing, and non-personalised methods of recommendation (Weighted Ratings using Normal-Inverse-Gamma distribution for top-n rankings, and Product Association-Driven systems)
- [systems/collabfiltering.ipynb](github.com/roodriigoooo/recsysreview/tree/main/systems/collabfiltering.ipynb) includes a brief explanation of common similarity measures, and explores user-based and item-based collaborative filtering, matrix factorization and summarizes their results. 
