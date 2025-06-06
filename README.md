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

We conducted permutation tests using the Kolmogorov-Smirnov (KS) statistic to assess whether missingness in avg_rating is dependent on other variables. The KS statistic measures the maximum distance between two empirical distribution functions.

***n_steps***

Results: KS Statistic: 0.0759, p-value: 0.000

We hypothesized that recipe complexity (measured by number of steps) might influence rating missingness because:

- Complex recipes may discourage users from completing and rating them
- Poorly rated recipes might correlate with complex preparation
- Users may be less likely to finish multi-step recipes

***protein_PDV***

Results: KS Statistic: 0.0175, p-value: 0.338

We selected protein content as a likely independent variable because:

- Nutritional content seems unrelated to user rating behavior
- Protein percentage is an intrinsic recipe property
- No plausible mechanism connects protein content to rating likelihood

With this, we believe that avg_rating is MAR based on protein_PDV.


### Hypothesis Testing

**Null Hypothesis:** There is no relationship for recipes with ease indicator tags and their average rating.

**Alternate Hypothesis:** Recipes with ease indicator tag have higher average ratings. with a permutation test.

**Test Statistic:** Mean ratings of recipes with ease indicator tags - mean ratings of recipes without ease indicator tags.

**Significance Value:** 0.05

**Result:** p-value = 0.009

We conducted a hypothesis test that looked at the relationship between the presence of an 'easy' tag and the average ratings. The ease indicator tags that we chose were the followng: 'easy', 'beginner', 'beginner-cook', '5-ingredients-or-less', '3-steps-or-less', '15-minutes-or-less', 'weeknight' because each of them was either the lowest '-or-less' tier or related to ease/beginner skill.

Difference in group means was an appropriate test statistic because it quantifies the strength of association between a binary tag variable and a continuous rating variable. Our resulting p-value was 0.008 which is less than 0.05 so we reject our null hypothesis. There is significant statistical evidence to support a difference in average ratings between recipes with and without the ease indicator tags.

ADD MORE ON TESTING PROCEDURE + PLOT


### Framing a Prediction Problem

We are addressing a regression problem to predict recipe cooking length. The response variale that we are predicting is minutes, or number of minutes it takes to finish cooking. We chose this because we thought it would be interesting to look at how different factors of a recipe might affect its cooking time. We evaluate the model using R², as it effectively measures how well the model explains the variance in cooking length, which is suitable for regression tasks. 


### Baseline Model

We used a Linear Regression model to predict recipe cooking length. The model used two quantitative features: the number of ingredients and the number of steps. Since no ordinal or nominal features were included, encoding was not necessary. The model's performance, measured by R² on both training and test sets, was weak (Train R²: 0.176, Test R²: 0.177), indicating limited explanatory power. We do not believe this model is 'good' enough because it fails to capture enough variation in cooking length, likely due to the simplicity and limited scope of the features used. 


### Final Model
To improve the model, we added four features: 'is_easy' to represent ease based on ease-relatd tags, 'avg_ing_len' to represent the average length of the ingredients, 'desc_length' to represent the length of the description, and 'meal_type' from the tags. These features aim to capture more nuanced information related to cooking time that the current features do not represent. Usually, 'is_easy' would indicate an easier recipe, and one way to measure or represent easy recipes is through a short cook time. Recipes with longer ingredient lengths might also indicate that they are more complex and take a longer time to finish, and the same goes for longer description lengths. Different meal types, such as breakfast, lunch, and snack might take shorter times to make because of culture and/or time-of-day eaten compared to the work day, whereas dinner might take longer because as a general idea (at least culturally in the US), people eat larger dinners, and fewer people are on a set timeframe for dinners.

We ended up using RandomForestRegressor
### Fairness Analysis
