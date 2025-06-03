# Culinary Computing
## Recipe Analysis and Prediction through Data Science and Machine Learning
## Names: Amber Tang and Maggie Zhang

### Introduction
The dataset we chose to analyze was the "Recipes and Ratings" dataset, which included data on various recipes and their ratings from food.com. The data comes from two csv files, one which includes recipes data and the other of which includes the reviews and ratings submitted for each recipe. We wanted to use these datasets to answer the question "Is there a significant relationship between recipe complexity and recipe rating?". We defined complexity as a combination of the variables: (standardized) minutes, number of ingredients and number of steps. To access this data, we first left merged the two datasets on recipe, then calculated the average rating per recipe and added it to the original "recipes" dataframe. Our final dataframe had 83782 rows (meaning 83782 recipes). The columns that we analyzed were "minutes" (Minutes to prepare recipe), "n_steps" (Number of steps in recipe) and "ingredients" (List of ingredients in recipe).

### Data Cleaning and Exploratory Data Analysis

### Assessment of Missingness

### Hypothesis Testing

### Framing a Prediction Problem

### Baseline Model

### Final Model

### Fairness Analysis
