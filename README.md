This is my take on the classic Home Credit Default Risk challenge from Kaggle. Even though the dataset is from 2018, I wanted to use it to show a different approach. Instead of just stacking 50 black-box models to get the highest ROC-AUC possible, I wanted to actually explain why people default, combining my background in economics with data science.
What is this project about?
This notebook covers the whole pipeline, from raw data to business decisions:

Data Wrangling: Using SQL and Pandas to filter, aggregate, and merge the historical credit tables. (And yes, you might see that I forgot to put index=False a couple of times while exporting CSVs. Lesson learned!).

Feature Selection: I let an XGBoost model do the heavy lifting to find the most predictive variables using its Gain score, avoiding manual bias.

The "Meta-Goat" Model: Here is where the econometrics kicks in. I used Weight of Evidence (WoE) to handle non-linearity, and a Logistic Regression as the final model so I could actually interpret the coefficients and check for collinearity (VIF).

Stacking OOF: To give the model a predictive boost without breaking strict statistical assumptions, I used Stratified K-Fold to create an Out-Of-Fold XGBoost probability and fed it to the Logit model as a new feature.

Business Insights: I calculated the Average Marginal Effects (AME) to translate the math into real business language (e.g., "Having higher education reduces default probability by 1.31% on average").
Why this approach?
As an economics student, I care about causality and explainability. In the real world, a bank can't reject a client just because "a neural network said so." By using a Logistic Regression powered by Machine Learning feature engineering, we get a solid predictive score (ROC-AUC ~0.745) but we keep the model 100% transparent and useful for the business side.
Note on the Visualizations
I originally used Plotly for the Exploratory Data Analysis (EDA) because I like the interactive graphs and custom themes. However, since the interactive file was too large for GitHub to render properly, I forced the notebook to render the final charts as static PNGs.

Feel free to dive into the notebook to see the code, the math, and the final business conclusions!
