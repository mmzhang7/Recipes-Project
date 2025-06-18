## Names: Amber Tang and Maggie Zhang

### Introduction

The dataset we chose to analyze is the **"Recipes and Ratings"** dataset from Food.com, which contains detailed information about various recipes alongside user-submitted reviews and ratings. The data comes from two CSV files: one containing metadata about each recipe and the other containing user interactions.

The central question guiding our analysis is: **What factors best predict the time it takes to complete a recipe?** Specifically, we aim to investigate how features such as the number of ingredients, number of steps, user ratings, recipe tags, and descriptions relate to the total preparation time.

To explore this question, we merged the recipe and rating datasets using a left join on the recipe ID and calculated the **average rating per recipe**, which we added to the original recipes dataframe. Our final dataset contains **83,782 recipes**.

**Relevant Columns**

From the recipes dataset:

- **`id`**: Unique ID for each recipe.
- **`name`**: The name of the recipe.
- **`minutes`**: Time (in minutes) required to prepare the recipe — this is our outcome variable.
- **`tags`**: Food.com tags for each recipe (e.g., "easy", "vegan", "holiday").
- **`n_steps`**: Number of steps in the recipe's instructions.
- **`steps`**: Text for each step of the recipe, in order.
- **`ingredients`**: A list of ingredients used in the recipe.
- **`description`**: User-provided description or blurb about the recipe.

From the ratings dataset:

- **`recipe_id`**: Corresponds to the recipe ID in the main dataset.
- **`rating`**: Rating given by a user (used to compute average rating per recipe).

We derived an additional feature:

- **`avg_rating`**: The mean of all ratings given to each recipe.

This dataset is particularly valuable to readers interested in food, time management, and data-driven cooking recommendations. By understanding which recipe features are most strongly associated with preparation time, we can help users make more informed choices about what to cook based on their time constraints and preferences.


### Data Cleaning and Exploratory Data Analysis

**Data Cleaning**

To ensure accurate and meaningful analysis, we performed several data cleaning steps based on our understanding of how the dataset was generated. Below, we outline each step, explain its rationale, and describe its impact on our analyses.

**1. Replacing 0-Star Ratings with Missing Values**

The first cleaning step involved addressing inconsistencies in the ratings data. We observed that some recipes had a rating value of **0**, which is not a valid user rating on Food.com. The minimum allowed rating on the platform is **1 star**. A rating of 0 typically appears when a user submits a review without assigning a star rating. To prevent these from skewing the computed average ratings downward, we replaced all 0s in the `rating` column with `np.nan`. This allowed us to exclude them when calculating the `average_rating` per recipe.

**Effect on analysis:** By excluding invalid ratings, we ensured that the average rating reflects actual user evaluations rather than missing data.

---

**2. Splitting the `nutrition` Column into Separate Features**

The `nutrition` column originally stored nutritional data as a single string in the format: [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]
This format limited our ability to analyze or model individual nutritional components. To address this, we created a function that parsed the string and extracted each value into its own numeric column. We added the following new columns:

- `calories`
- `total_fat_PDV`
- `sugar_PDV`
- `sodium_PDV`
- `protein_PDV`
- `saturated_fat_PDV`
- `carbohydrates_PDV`

**Effect on analysis:** This transformation allowed us to include specific nutritional features in modeling and correlation analyses, improving the interpretability and granularity of our results.

---

**3. Parsing List-Formatted Strings into Actual Python Lists**

The `tags`, `steps`, and `ingredients` columns were stored as strings that resembled Python lists (e.g., `"['easy', 'vegan']"`). These needed to be converted into actual list objects to support proper manipulation and analysis.

We implemented a function, `parse_list_str`, which:
- Removed leading/trailing brackets and quotation marks,
- Split the string into elements,
- Returned a clean list.

We applied this function to the following columns:
- `tags`
- `steps`
- `ingredients`

**Effect on analysis:** This step enabled us to count elements (e.g., number of ingredients), identify recipes with certain tags, and perform keyword-based filtering or encoding.

---

**Final Cleaned Data Preview**

Below is a preview of the cleaned dataset (`recipes.head()`), including the processed columns:

|     id | name                                 |   minutes | tags                                                                                                                                                                                                                                                                                               |   n_steps | steps                                       | ingredients                                 | description                                 |   avg_rating |
|-------:|:-------------------------------------|----------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:--------------------------------------------|:--------------------------------------------|:--------------------------------------------|-------------:|
| 333281 | 1 brownies in the world    best ever |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        10 | ['heat the oven to 350f and arrange the ... | ['bittersweet chocolate', 'unsalted butt... | these are the most; chocolatey, moist, r... |            4 |
| 453467 | 1 in canada chocolate chip cookies   |        45 | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        12 | ['pre-heat oven the 350 degrees f', 'in ... | ['white sugar', 'brown sugar', 'salt', '... | this is the recipe that we use at my sch... |            5 |
| 306168 | 412 broccoli casserole               |        40 | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |         6 | ['preheat oven to 350 degrees', 'spray a... | ['frozen broccoli cuts', 'cream of chick... | since there are already 411 recipes for ... |            5 |
| 286009 | millionaire pound cake               |       120 | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |         7 | ['freheat the oven to 300 degrees', 'gre... | ['butter', 'sugar', 'eggs', 'all-purpose... | why a millionaire pound cake?  because i... |            5 |
| 475785 | 2000 meatloaf                        |        90 | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        17 | ['pan fry bacon', 'and set aside on a pa... | ['meatloaf mixture', 'unsmoked bacon', '... | ready, set, cook! special edition contes... |            5 |

**Univariate Analysis: Number of Ingredients**

One of the variables we decided to examine was the **number of ingredients**, as we believed it could serve as a useful proxy for recipe complexity. The histogram below shows the distribution:

<iframe
  src="n_ingredients_distribution.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The distribution is right-skewed. Most recipes contain between **5 to 10 ingredients**, suggesting that simpler recipes are more common on Food.com. However, there are a few outliers with up to **36 ingredients**, indicating the presence of more complex recipes. This variability in ingredient count gives us a way to explore how recipe complexity may impact user ratings.

---

**Bivariate Analysis: Ingredients vs. Average Rating**

To further explore the role of complexity, we analyzed the relationship between **number of ingredients** and **average user rating**. The scatterplot below visualizes this relationship:

<iframe
  src="ingredients_vs_rating.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The scatterplot reveals that recipes with **fewer ingredients** exhibit a wider range of average ratings, including more low-rated outliers. In contrast, recipes with **many ingredients** tend to have consistently high ratings, although they are less common. This suggests that while simple recipes are accessible and popular, more **ingredient-rich recipes may be perceived as higher quality or more reliably satisfying** by users.

---



**Interesting Aggregates: Number of Ingredients vs. Rating**

To further explore the relationship between recipe complexity and user satisfaction, we aggregated our data by the **`n_ingredients`** column and created a pivot table that summarizes:

- **`count`**: Number of recipes with that ingredient count  
- **`mean`**: Average rating  
- **`median`**: Median rating  

The table below displays these values:

| n_ingredients | count | mean   | median |
|---------------|-------|--------|--------|
| 1             | 13    | 4.8615 | 5      |
| 2             | 723   | 4.6926 | 5      |
| 3             | 2,280 | 4.6620 | 5      |
| 4             | 4,348 | 4.6339 | 5      |
| 5             | 6,355 | 4.6474 | 5      |
| 6             | 7,302 | 4.6331 | 5      |
| 7             | 8,271 | 4.6241 | 5      |
| 8             | 8,657 | 4.6115 | 5      |
| 9             | 8,378 | 4.6064 | 5      |
| 10            | 7,765 | 4.6102 | 5      |
| 11            | 6,751 | 4.6228 | 5      |
| 12            | 5,557 | 4.6168 | 5      |
| 13            | 4,353 | 4.6318 | 5      |
| 14            | 3,126 | 4.6164 | 5      |
| 15            | 2,316 | 4.6306 | 5      |
| 16            | 1,629 | 4.6250 | 5      |
| 17            | 1,104 | 4.6326 | 5      |
| 18            | 754   | 4.6876 | 5      |
| 19            | 489   | 4.6116 | 5      |
| 20            | 357   | 4.6075 | 5      |
| 21            | 209   | 4.6591 | 5      |
| 22            | 136   | 4.6934 | 5      |
| 23            | 90    | 4.7785 | 5      |
| 24            | 72    | 4.6047 | 5      |
| 25            | 37    | 4.7072 | 5      |
| 26            | 28    | 4.7593 | 5      |
| 27            | 23    | 4.6097 | 5      |
| 28            | 17    | 4.8592 | 5      |
| 29            | 10    | 4.9657 | 5      |
| 30            | 11    | 4.8682 | 5      |
| 31            | 8     | 5.0000 | 5      |
| 32            | 2     | 5.0000 | 5      |
| 33            | 1     | 5.0000 | 5      |
| 37            | 1     | 5.0000 | 5      |

This table confirms several trends observed earlier. Most recipes cluster between **5–10 ingredients**, reflecting the popularity of simpler dishes. While **average ratings are generally high** across all groups, an intriguing pattern appears: **recipes with more ingredients (31–37) consistently receive perfect 5-star ratings**, although these are rare. This may indicate that more elaborate recipes, while less common, are especially appreciated by users, leading to greater satisfaction. However, the lack of variation in median ratings (all are 5) also suggests that **user ratings may not be very sensitive to ingredient count**.

---

### Assessment of Missingness

**NMAR Analysis**

We believe the avg_rating column in the dataset is NMAR. The missing values occur because some recipes have not been rated by users, and this missingness may depend on unobserved factors such as the recipe’s popularity or visibility on the platform. For example, less appealing or rarely viewed recipes may be less likely to receive ratings. To potentially make this missingness MAR, we could collect additional data such as the number of views, saves, or shares a recipe has, or whether it includes a photo or tags.

**Missingness Dependency**

We conducted permutation tests using the Kolmogorov-Smirnov (KS) statistic to assess whether missingness in avg_rating is dependent on other variables. The KS statistic measures the maximum distance between two empirical distribution functions.

***n_steps***

**Null Hypothesis:** The distribution of `n_steps` when `avg_rating` is missing is the same as the distribution of `n_steps` when `avg_rating` is not missing.

**Alternate Hypothesis:** The distribution of `n_steps` when `avg_rating` is missing is not same as the distribution of `n_steps` when `avg_rating` is not missing.

**Test Statistic:** K-S Statistic

**Significance Level:** 0.05

**Results:** Observed KS Statistic: 0.076

**p-value:** 0.000

We hypothesized that recipe complexity (measured by number of steps) might influence rating missingness because:

- Complex recipes may discourage users from completing and rating them
- Poorly rated recipes might correlate with complex preparation
- Users may be less likely to finish multi-step recipes

***protein_PDV***

**Null Hypothesis:** The distribution of `protein_PDV` when `avg_rating` is missing is the same as the distribution of `protein_PDV` when `avg_rating` is not missing.

**Alternate Hypothesis:** The distribution of `protein_PDV` when `avg_rating` is missing is not same as the distribution of `protein_PDV` when `avg_rating` is not missing.

**Test Statistic:** K-S Statistic

**Significance Level:** 0.05

**Results:** Observed KS Statistic: 0.018

**p-value:** 0.324

We selected protein content as a likely independent variable because:

- Nutritional content seems unrelated to user rating behavior
- Protein percentage is an intrinsic recipe property
- No plausible mechanism connects protein content to rating likelihood

The likelihood of a rating being missing appears to depend on recipe complexity (n_steps), but not on nutritional content (protein_PDV). This implies that unobserved factors like user engagement, recipe appeal, or difficulty may drive the missingness, and collecting metadata such as views, saves, or image presence could help model it better.

<iframe
  src="ks.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The empirical distribution of the Kolmogorov-Smirnov (K-S) statistic shown in the plot represents the expected variation in distributional differences between the 'n_steps' variable for recipes with and without missing 'avg_rating' values under the null hypothesis of randomness. The histogram, based on 1,000 random permutations, indicates the distribution of K-S statistics we would expect by chance. The red vertical line marks the observed K-S statistic (0.08), which lies in the upper tail of the distribution. This suggests that the observed difference in 'n_steps' between the two groups is larger than what would typically occur due to random variation alone. Consequently, there is evidence that the missingness in 'avg_rating' may be systematically related to the number of steps in a recipe, potentially violating the assumption of missing completely at random (MCAR). To show MAR, we might need data on something like recipe views to test whether missing ratings depend on visibility.

### Hypothesis Testing

**Null Hypothesis:** There is no relationship for recipes with ease indicator tags and their average rating.

**Alternate Hypothesis:** Recipes with ease indicator tag have higher average ratings. with a permutation test.

**Test Statistic:** Mean ratings of recipes with ease indicator tags - mean ratings of recipes without ease indicator tags.

**Significance Level:** 0.05

**Result:** p-value = 0.007

We conducted a hypothesis test that looked at the relationship between the presence of an 'easy' tag and the average ratings. The ease indicator tags that we chose were the followng: 'easy', 'beginner', 'beginner-cook', '5-ingredients-or-less', '3-steps-or-less', '15-minutes-or-less', 'weeknight' because each of them was either the lowest '-or-less' tier or related to ease/beginner skill. These were selected by keyword search.

Difference in group means was an appropriate test statistic because it quantifies the strength of association between a binary tag variable and a continuous rating variable. Our resulting p-value was 0.008 which is less than 0.05, the most common significance level, so we reject our null hypothesis. There is significant statistical evidence to support a difference in average ratings between recipes with and without the ease-related tags.

<iframe
  src="means.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

The histogram shows the range of mean differences expected by chance under the null hypothesis that there is no real difference in ratings between the two groups. The red vertical line represents the observed difference of 0.011, which lies in the far-right tail of the distribution. This indicates that the observed difference is larger than most (in the case of this specific permutation test, all) differences generated under the null hypothesis.


### Framing a Prediction Problem

We are addressing a regression problem to predict recipe cooking length. We chose to predict minutes of cooking length because it seems that there are a lot of possible factors that could influence it. For our baseline model, we plan to use the three features: the number of steps, number of ingredients and average rating. We will apply Linear Regression and evaluate the model using R², as it effectively measures how well the model explains the variance in cooking length, which is suitable for regression tasks.  


### Baseline Model

For our baseline model, we built two linear regression models to predict recipe preparation time (measured in minutes) and compared them to a constant model. Our Simple Linear Regression Model used a single predictor (n_steps) and our Multiple Linear Regression Model used two predictors (n_steps and n_ingredients).

| Feature         | Type         | Description                                      |
| --------------- | ------------ | ------------------------------------------------ |
| `n_steps`       | Quantitative | Number of steps in the recipe                    |
| `n_ingredients` | Quantitative | Number of ingredients used                       |

No nominal or ordinal variables were included, so no encoding was necessary. However, for the multiple linear regression model, all quantitative features were converted into polynomial features using PolynomialFeatures from sklearn. This is because the regression plots show a non-linear relationship between minutes and our predictors/features. For minutes, since there were a few outliers and the distribution of it did not look anything close to roughly normal, we used up to the 99th percentile and then log-transformed it to reduce skew. For our Test and Training split, we used 80% Train and 20% Test for a good balance of enough training and enough testing data.

The following table displays the RMSE and R² Score from each of these models.

| Model Type                                                            | RMSE    | R² Score  |
| --------------------------------------------------------------------- | ------- | --------- |
| Constant Baseline                                                     | 4020.64 | N/A       |
| Simple Linear Regression (`n_steps`)                                  | 4020.53 | 0.000055  |
| Multiple Linear Regression (`n_steps`, `n_ingredients`)               | 4020.24 | 0.218     |

The residual plot for the simple linear model is displayed below:
<iframe
  src="regression.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>

Our Multiple Linear Regression baseline model produced a Train R² value of 0.215 and a Test R² value of 0.218. Therefore, no, we do not believe the current model is a “good” one. While it was implemented correctly and meets the assumptions of linear regression on the surface, its predictive power is extremely low. The features used explain almost none of the variation in cooking time, as reflected in the very low R² scores; only about 22% of the variance can be explained. Our plans for improving on the model include adding different transformations, including one-hot encodings for some speed keywords in the tags.

Overall, while the model offers a basic start, its performance metrics indicate that it is not suitable for making accurate predictions in its current form.


### Final Model

**Feature Engineering and Rationale**

Several custom features were engineered to better capture meaningful characteristics of the recipes for predicting preparation time:

| Feature Name         | Type         | Description                                                                 |
|----------------------|--------------|-----------------------------------------------------------------------------|
| `is_easy`            | Binary       | Indicates whether a recipe is tagged as "easy" based on common tag labels. |
| `avg_ingredient_len` | Quantitative | Average word count per ingredient, a proxy for ingredient complexity.       |
| `desc_length`        | Quantitative | Number of words in the recipe description, indicating instructional depth.  |
| `meal_type`          | Nominal      | Meal category (e.g., breakfast, lunch) extracted from tags.                 |

These features relate directly to the recipe’s nature and complexity, which intuitively should impact cooking time and improve the model’s predictive ability.

We also included a polynomial transformation (degree=2) on `n_ingredients` and `n_steps` to capture potential nonlinear effects and interactions between these core numeric variables.

**Modeling Approach and Pipeline**

We used a **Random Forest Regressor**, chosen for its ability to model nonlinear relationships, handle both numeric and categorical variables, and provide strong performance with minimal tuning. The model was implemented within a `Pipeline` that included:

- Custom feature engineering using `FunctionTransformer`s to compute `is_easy`, `avg_ingredient_len`, `desc_length`, and `meal_type`,
- Polynomial expansion of `n_ingredients` and `n_steps` to capture interaction effects,
- One-hot encoding of categorical features like `meal_type`,
- A final `RandomForestRegressor()` using default hyperparameters.

**Model Performance**

The original model performance was:

- **Train R² = 0.837**
- **Test R² = 0.126**

This significant train–test gap indicated **overfitting**, where the model fit the training data well but failed to generalize to unseen examples.

Using **GridSearchCV**, we finetuned the polyomial hyperparameters, 

- Best Parameters: {'poly_features__degree': 2, 'poly_features__interaction_only': False}
- **Test R² ≈ 0.274**

This reflects more than a twofold improvement in generalization (R²) and a reduction in average prediction error after hyperparameter finetuning. These results support our intuition that features reflecting recipe complexity, structure, and context contribute meaningfully to more accurate cooking time predictions.

### Fairness Analysis

**Group Definitions**

Recipes are divided into two groups based on the number of ingredients:
- **Group X (Short Recipes):** Recipes with fewer than 8 ingredients.
- **Group Y (Long Recipes):** Recipes with 8 or more ingredients.

This threshold was chosen as a simple, interpretable cutoff to examine whether model performance differs based on recipe complexity.

**Evaluation Metric**

The evaluation metric used to compare model performance across groups is the **R² score**. This metric measures the proportion of variance in the target variable (`minutes`) that is predictable from the input features.

**Hypotheses**

We aim to assess whether the model performs equally well for both short and long recipes. To do this, we frame the following hypotheses:

- **Null Hypothesis (H₀):** There is no difference in R² performance between Group X and Group Y. Any observed difference is due to chance.
- **Alternative Hypothesis (H₁):** There is a significant difference in R² performance between Group X and Group Y.

**Test Statistic:** the difference in R² scores between the two groups: **ΔR² = 0.241 − 0.136 = 0.105**
This positive difference indicates that the model performs substantially better on short recipes than on long ones at the current distribution.

**Testing Procedure**

To assess the significance of the observed difference, we used a **permutation test**:
- We shuffled the group labels (short vs. long) 500 times while keeping the features and targets fixed.
- For each shuffle, we re-calculated the difference in R² scores between the pseudo-groups.
- We then compared the observed difference to this null distribution.

**Significance Level**

We used a significance level of **0.05** to evaluate the p-value.
- The resulting **p-value = 0.00**, indicating that the observed difference is highly unlikely to have occurred by chance.

**Conclusion**

Since the **p-value (0.00) < α (0.05)**, we **reject the null hypothesis**.  
This provides strong evidence that the model performs significantly better on short recipes than on long recipes, highlighting a potential **fairness concern** in model performance across recipe complexity.

The histogram below shows the empirical distribution of the difference in R² scores between short and long recipes under the null hypothesis, generated via permutation testing. Most of the simulated differences cluster tightly around zero, indicating that under random group assignments, the R² difference is typically very small. The observed difference of 0.105, marked by the red vertical line, lies far to the right of this distribution. This separation suggests that the observed performance gap is highly unlikely to have occurred by chance, providing strong evidence that the model genuinely performs better on short recipes than long ones.

<iframe
  src="r^2 distribution for fairness.html"
  width="600"
  height="600"
  frameborder="0"
></iframe>
