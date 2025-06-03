# Culinary Computing
## Recipe Analysis and Prediction through Data Science and Machine Learning
## Names: Amber Tang and Maggie Zhang

### Introduction

The dataset we chose to analyze was the "Recipes and Ratings" dataset, which included data on various recipes and their ratings from food.com. The data comes from two csv files, one which includes recipes data and the other of which includes the reviews and ratings submitted for each recipe. We wanted to use these datasets to answer the question "Is there a significant relationship between recipe complexity and recipe rating?". We defined complexity as a combination of the variables: (standardized) minutes, number of ingredients and number of steps. To access this data, we first left merged the two datasets on recipe, then calculated the average rating per recipe and added it to the original "recipes" dataframe. Our final dataframe had 83782 rows (meaning 83782 recipes). The columns that we analyzed were "minutes" (Minutes to prepare recipe), "n_steps" (Number of steps in recipe) and "ingredients" (List of ingredients in recipe).

### Data Cleaning and Exploratory Data Analysis

**Data Cleaning**

The first step we took to clean our data was filling ratings of 0 with np.nan. This is because these 0 ratings are from reviews where the reviewer failed to add a rating. The minimum possible star rating to give a recipe on food.com is a 1 star, and it only displays as 0 stars when a rating is missing. We added the np.nan values instead so the average rating is not incorrectly lowered.

The next step we took to clean our data was to split up the nutrition column into multiple columns for each variable contained in it. Each entry in this column was a string in the format:[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]. We implemented a function that parsed through each string, separated it into the different variables present and then added the separate values in at individual columns. We ended up adding these following columns: calories, total_fat_PDV, sugar_PDV, sodium_PDV,protein_PDV,saturated_fat_PDV and carbohydrates_PDV.

Additionally, we noticed that there were other columns with a similar list format. The tags, steps and ingredients columns were strings that looked like lists. We implemented a parse_list_str function that took a string and returned it in a list. We applied this function to the tags, steps and ingredients columns and replaced the strings in these columns with their list equivalents. Our final cleaned dataframe head is displayed below:

![Head](df_head.png)

**Univariate Analysis**

One of the variables we decided to look at was the number of ingredients since we thought it could be a good indicator of complexity. The histogram generated is displayed below:
<iframe
  src="n_ingredients_distribution.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The distribution of the number of ingredients is skewed. For most recipes, the number of ingredients falls around 5-10 ingredients. There are a few outlier swith a maximum of 36 ingredients included.

**Bivariate Analysis**

We continued to analyze our data by examining the relationship between the number of ingredients and the average rating. We created the scatterplot below:
<iframe
  src="ingredients_vs_rating.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

From the scatterplot we can see that most recipes, regardless of ingredient count, receive high average ratings close to 5. However, simpler recipes show a wider range of ratings, including more low-rated outliers. In contrast, recipes with more ingredients tend to receive consistently high ratings, though they are less frequent. This suggests that while simple recipes are popular, more complex ones may be more reliably well-received.

**Interesting Aggregates**

We chose to aggregate our data by the "n_ingredients" column and looked at the count, mean and median values. The pivot table generated is shown below:

|   n_ingredients |   count |    mean |   median |
|----------------:|--------:|--------:|---------:|
|               1 |      13 | 4.86154 |        5 |
|               2 |     723 | 4.69258 |        5 |
|               3 |    2280 | 4.66203 |        5 |
|               4 |    4348 | 4.63394 |        5 |
|               5 |    6355 | 4.64743 |        5 |
|               6 |    7302 | 4.6331  |        5 |
|               7 |    8271 | 4.62407 |        5 |
|               8 |    8657 | 4.61152 |        5 |
|               9 |    8378 | 4.60638 |        5 |
|              10 |    7765 | 4.61023 |        5 |
|              11 |    6751 | 4.62276 |        5 |
|              12 |    5557 | 4.61677 |        5 |
|              13 |    4353 | 4.63185 |        5 |
|              14 |    3126 | 4.61637 |        5 |
|              15 |    2316 | 4.6306  |        5 |
|              16 |    1629 | 4.62501 |        5 |
|              17 |    1104 | 4.63257 |        5 |
|              18 |     754 | 4.68762 |        5 |
|              19 |     489 | 4.61161 |        5 |
|              20 |     357 | 4.60754 |        5 |
|              21 |     209 | 4.65912 |        5 |
|              22 |     136 | 4.69335 |        5 |
|              23 |      90 | 4.77853 |        5 |
|              24 |      72 | 4.60467 |        5 |
|              25 |      37 | 4.70721 |        5 |
|              26 |      28 | 4.7593  |        5 |
|              27 |      23 | 4.60973 |        5 |
|              28 |      17 | 4.85924 |        5 |
|              29 |      10 | 4.96571 |        5 |
|              30 |      11 | 4.86818 |        5 |
|              31 |       8 | 5       |        5 |
|              32 |       2 | 5       |        5 |
|              33 |       1 | 5       |        5 |
|              37 |       1 | 5       |        5 |

The pivot table shows that recipes with fewer ingredients are much more common, with the highest counts between 5 and 10 ingredients. Average ratings are generally high across all ingredient counts, with all medians staying at 5.0. An interesting observation is that the recipes with the highest number of ingredients (31-37) received perfect 5 star ratings in all reviews. Overall, there’s no clear link between the number of ingredients and average rating since simpler and more complex recipes both tend to receive high ratings.

### Assessment of Missingness

**NMAR Analysis**

We believe the avg_rating column in the dataset is NMAR. The missing values occur because some recipes have not been rated by users, and this missingness may depend on unobserved factors such as the recipe’s popularity or visibility on the platform. For example, less appealing or rarely viewed recipes may be less likely to receive ratings. To potentially make this missingness MAR, we could collect additional data such as the number of views, saves, or shares a recipe has, or whether it includes a photo or tags—factors that might influence whether a user rates a recipe.

**Missingness Dependency**

TO BE WRITTEN

### Hypothesis Testing

**Null Hypothesis:** There is no relationship for recipes tagged with the 'easy' tag and its average rating.

**Alternate Hypothesis:** Recipes tagged with the 'easy' tag have higher average ratings.
with a permutation test.

**Test Statistic:** Pearson's r

**Significance Value**: 0.05

**Result:** p-value = 0.27

We conducted a hypothesis test that looked at the relationship between the presence of an 'easy' tag and the average ratings. Pearson’s r was an appropriate test statistic because it quantifies the strength of association between a binary tag variable and a continuous rating variable. Our resulting p-value was 0.27 which is greater than 0.5 so we fail to reject null hypothesis. We do not have significant evidence of a difference in average ratings between recipes with and without the 'easy' tags.

ADD MORE ON TESTING PROCEDURE + PLOT


### Framing a Prediction Problem

We are addressing a regression problem to predict recipe cooking length. Our baseline model uses two features available at the time of prediction: the number of ingredients and the number of steps. We apply Linear Regression and evaluate the model using R², as it effectively measures how well the model explains the variance in cooking length, which is suitable for regression tasks. 


### Baseline Model

We used a Linear Regression model to predict recipe cooking length. The model used two quantitative features: the number of ingredients and the number of steps. Since no ordinal or nominal features were included, encoding was not necessary. The model's performance, measured by R² on both training and test sets, was weak indicating limited explanatory power. We do not believe this model is 'good' enough because it fails to capture enough variation in cooking length, likely due to the simplicity and limited scope of the features used. To improve the model, we plan to incorporate additional features such as tf-idf vectors for speed-related words in the recipe description and one-hot encodings of relevant speed keywords in tags. These features aim to capture more nuanced information related to cooking time that the current features do not represent.


### Final Model

### Fairness Analysis
