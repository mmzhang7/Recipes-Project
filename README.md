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

Results: KS Statistic: 0.76, p-value: 0.000

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

The likelihood of a rating being missing appears to depend on recipe complexity (n_steps), but not on nutritional content (protein_PDV). This implies that unobserved factors like user engagement, recipe appeal, or difficulty may drive the missingness, and collecting metadata such as views, saves, or image presence could help model it better.

<iframe
  src="n_steps_ks.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The empirical distribution of the Kolmogorov-Smirnov (K-S) statistic shown in the plot represents the expected variation in distributional differences between the 'n_steps' variable for recipes with and without missing 'avg_rating' values under the null hypothesis of randomness. The histogram, based on 1,000 random permutations, indicates the distribution of K-S statistics we would expect by chance. The red vertical line marks the observed K-S statistic (0.08), which lies in the upper tail of the distribution. This suggests that the observed difference in 'n_steps' between the two groups is larger than what would typically occur due to random variation alone. Consequently, there is evidence that the missingness in 'avg_rating' may be systematically related to the number of steps in a recipe, potentially violating the assumption of missing completely at random (MCAR).

### Hypothesis Testing

**Null Hypothesis:** There is no relationship for recipes with ease indicator tags and their average rating.

**Alternate Hypothesis:** Recipes with ease indicator tag have higher average ratings. with a permutation test.

**Test Statistic:** Mean ratings of recipes with ease indicator tags - mean ratings of recipes without ease indicator tags.

**Significance Value:** 0.05

**Result:** p-value = 0.008

We conducted a hypothesis test that looked at the relationship between the presence of an 'easy' tag and the average ratings. The ease indicator tags that we chose were the followng: 'easy', 'beginner', 'beginner-cook', '5-ingredients-or-less', '3-steps-or-less', '15-minutes-or-less', 'weeknight' because each of them was either the lowest '-or-less' tier or related to ease/beginner skill.

Difference in group means was an appropriate test statistic because it quantifies the strength of association between a binary tag variable and a continuous rating variable. Our resulting p-value was 0.008 which is less than 0.05 so we reject our null hypothesis. There is significant statistical evidence to support a difference in average ratings between recipes with and without the ease 

<iframe
  src="hypothesis_test.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The histogram shows the range of mean differences expected by chance under the null hypothesis that there is no real difference in ratings between the two groups. The red vertical line represents the observed difference of 0.011, which lies in the far-right tail of the distribution. This indicates that the observed difference is larger than most differences generated under the null hypothesis.


### Framing a Prediction Problem

We are addressing a regression problem to predict recipe cooking length. We chose to predict minutes of cooking length because it seems that there are a lot of possible factors that could influence it. For our baseline model, we plan to use the three features: the number of steps, number of ingredients and average rating. We will pply Linear Regression and evaluate the model using R², as it effectively measures how well the model explains the variance in cooking length, which is suitable for regression tasks.  


### Baseline Model

For our baseline model, we built two linear regression models to predict recipe preparation time (measured in minutes) and compared them to a constant model. Our Simple Linear Regression Model used a single predictor (n_steps) and our Multiple Linear Regression Model used three predictors (n_steps, n_ingredients and avg_rating).

| Feature         | Type         | Description                                      |
| --------------- | ------------ | ------------------------------------------------ |
| `n_steps`       | Quantitative | Number of steps in the recipe                    |
| `n_ingredients` | Quantitative | Number of ingredients used                       |
| `avg_rating`    | Quantitative | Average rating of the recipe (from user reviews) |

No nominal or ordinal variables were included, so no encoding was necessary. However, for the multiple linear regression model, all quantitative features were standardized using StandardScaler from sklearn to ensure comparability.

The following table displays the RMSE and R^2 Score from each of these models.

| Model Type                                                            | RMSE    | R² Score  |
| --------------------------------------------------------------------- | ------- | --------- |
| Constant Baseline                                                     | 4402.64 | N/A       |
| Simple Linear Regression (`n_steps`)                                  | 4402.53 | 0.000055  |
| Multiple Linear Regression (`n_steps`, `n_ingredients`, `avg_rating`) | 4402.34 | 0.0002817 |

The residual plot for the simple libnear model is displayed below:
<iframe
  src="residual_plot.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

No, we do not believe the current model is a “good” one. While it was implemented correctly and meets the assumptions of linear regression on the surface, its predictive power is extremely low. The features used explain almost none of the variation in cooking time, as reflected in the very low R² scores. Additionally, a residual plot of the simple linear model reveals residuals increasing for recipes with longer predicted times—suggesting a poor model fit. Our plans for improving on the model include using tf-idf for speed words in the description and/or one-hot vector encodings for some speed keywords in the tags.

Overall, while the model offers a basic start, its performance metrics indicate that it is not suitable for making accurate predictions in its current form.


### Final Model

To improve prediction of recipe preparation time, We created several new features based on domain knowledge and patterns in the data:
| Feature Name         | Type         | Description                                                                |
| -------------------- | ------------ | -------------------------------------------------------------------------- |
| `is_easy`            | Binary       | Indicates whether a recipe is tagged as "easy" based on common tag labels. |
| `avg_ingredient_len` | Quantitative | Average word count per ingredient, a proxy for ingredient complexity.      |
| `desc_length`        | Quantitative | Number of words in the recipe description, indicating instructional depth. |
| `meal_type`          | Nominal      | Meal category (e.g., breakfast, lunch) extracted from tags.                |

We chose these features because we believe that `is_easy` may reflect shorter recipes with fewer steps or simpler methods `avg_ingredient_len` captures how elaborate each ingredient is (e.g., “freshly grated parmesan” vs. “cheese”). `desc_length` correlates with complexity: detailed recipes often require more time. `meal_type` captures context: dinner recipes tend to take longer than snacks or breakfast.

These features align with the data-generating process: recipe time is likely influenced by the recipe’s intended difficulty, type, and complexity of instructions and ingredients.

The modeling algorithm used was a Random Forest Regressor, selected for its ability to model non-linear relationships, handle mixed feature types, and provide robustness against overfitting.

Hyperparameter tuning was conducted in two phases:

Grid Search was first performed over combinations of `n_estimators` (1000, 5000), `max_depth` (10, 15), and `max_features` ('sqrt'). The best result from this phase had 1000 trees, a max depth of 15, and used the square root of the number of features at each split.
### Fairness Analysis
