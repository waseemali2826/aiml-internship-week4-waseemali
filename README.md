STEP 18 — WRITTEN ANALYSIS REPORT
1. Executive Summary

This project focused on predicting house prices using supervised machine learning regression models on the Ames Housing dataset. Five models were trained and evaluated: Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, and a Polynomial + Ridge Pipeline. Extensive preprocessing and feature engineering were applied, including missing value handling, quality encoding, one-hot encoding, frequency encoding, scaling, and log transformation of the target variable. Among all models, the Polynomial + Ridge Pipeline achieved the best overall performance because it successfully captured nonlinear relationships while controlling overfitting through regularization. Evaluation metrics such as RMSE, R², Adjusted R², MAPE, cross-validation performance, and residual diagnostics were used to compare models comprehensively. The final model demonstrated strong predictive accuracy and stable generalization across folds, making it the best candidate for deployment in a real-world house price prediction system.

2. Feature Engineering Impact

Feature engineering played a critical role in improving model performance throughout the project. Quality-based features such as OverallQual, KitchenQual, and ExterQual became significantly more informative after ordinal encoding because the models could interpret their ranking numerically. Frequency encoding of Neighborhood helped preserve location importance while reducing dimensionality compared to one-hot encoding. One-hot encoding for nominal categorical features allowed the regression models to learn from property characteristics like house style, garage type, and exterior material. Log transformation of SalePrice reduced skewness and stabilized variance, improving linear model assumptions and reducing the impact of extreme values. Highly correlated features such as GrLivArea, GarageCars, TotalBsmtSF, and 1stFlrSF consistently appeared among the strongest predictors across nearly all models, confirming their importance in determining housing prices.

3. Model-by-Model Analysis
Linear Regression

Linear Regression served as the baseline model for the project. It was trained using standardized features and evaluated on both training and testing datasets. The model achieved relatively strong performance because many housing features have approximately linear relationships with price. However, the model struggled to fully capture nonlinear interactions between variables. The training and testing R² scores were reasonably close, indicating moderate generalization with limited overfitting. Despite this, residual analysis revealed patterns suggesting that purely linear assumptions were insufficient for modeling the dataset perfectly.

Ridge Regression

Ridge Regression was introduced to reduce overfitting and stabilize coefficients by applying L2 regularization. Multiple alpha values were explored manually and through GridSearchCV to identify the optimal regularization strength. Small alpha values behaved similarly to Linear Regression, while larger values increasingly shrank coefficients toward zero. Ridge improved stability and slightly improved test performance compared to the baseline model. The training and testing R² scores remained close, indicating good generalization. Ridge was especially useful because many housing features were correlated, and L2 regularization effectively handled multicollinearity without eliminating important predictors entirely.

Lasso Regression

Lasso Regression added L1 regularization, enabling both regularization and automatic feature selection. Different alpha values were tested to examine how aggressively the model eliminated coefficients. As alpha increased, more coefficients became exactly zero, reducing model complexity. Lasso successfully identified the most influential housing features while removing weaker predictors. However, excessive regularization caused underfitting at high alpha values because too many useful features were eliminated. The best Lasso model achieved competitive performance while using fewer features, improving interpretability. Features like OverallQual, GrLivArea, and GarageCars survived regularization, confirming their strong predictive importance.

ElasticNet

ElasticNet combined Ridge and Lasso penalties to balance coefficient shrinkage and feature selection simultaneously. A grid search over alpha and l1_ratio values was conducted to identify the best parameter combination. ElasticNet performed well because the housing dataset contained correlated features, which pure Lasso can struggle with. The model achieved a balance between sparsity and stability, often outperforming plain Lasso while remaining more interpretable than Ridge. The training and testing R² values showed reasonable generalization, although the performance improvement over Ridge was modest.

Polynomial + Ridge Pipeline

The Polynomial + Ridge Pipeline achieved the best overall results in the project. PolynomialFeatures enabled the model to capture nonlinear interactions between features, while Ridge regularization prevented the expanded feature space from causing severe overfitting. Degree 2 polynomial expansion significantly improved predictive accuracy, whereas higher degrees introduced excessive complexity and overfitting. The final pipeline achieved the highest R² and lowest RMSE among all models. Cross-validation confirmed strong generalization performance, and residual diagnostics showed improved residual distribution and reduced systematic patterns. This model provided the best balance between flexibility, predictive power, and stability.

4. Regularization Insights

Ridge, Lasso, and ElasticNet all use regularization to improve generalization, but they behave differently mathematically and practically. Ridge Regression uses L2 regularization, which shrinks coefficients continuously toward zero but rarely eliminates them entirely. This helps reduce variance and stabilize models with correlated predictors. Lasso Regression uses L1 regularization, which can drive coefficients exactly to zero, effectively performing automatic feature selection. ElasticNet combines both penalties, benefiting from Ridge’s stability and Lasso’s sparsity.

In this project, Lasso eliminated several weak or redundant features while preserving highly informative variables such as OverallQual, GrLivArea, GarageCars, and TotalBsmtSF. This made practical sense because these features directly relate to property quality, size, and functionality. ElasticNet provided a balanced solution by reducing overfitting while retaining correlated but informative predictors. Overall, Ridge achieved the most stable coefficients, Lasso produced the most interpretable model, and ElasticNet provided a compromise between both approaches.

5. Residual Analysis Findings

Residual diagnostics revealed important information about model quality and assumption validity. The Residuals vs Fitted plot for the best model showed mostly random scatter with limited visible patterns, indicating that the model captured much of the systematic structure in the data. The residual histogram appeared approximately bell-shaped, while the Q-Q plot showed most residual points following the diagonal line, suggesting approximate normality. The Scale-Location plot demonstrated relatively stable variance across fitted values, although slight heteroscedasticity remained for extreme house prices. The Shapiro-Wilk test provided additional statistical evidence regarding residual normality. Overall, the assumptions of linear regression were reasonably satisfied after preprocessing, feature engineering, and log transformation of the target variable.

6. Best Model Recommendation

The Polynomial + Ridge Pipeline is the recommended model for deployment because it achieved the best balance between predictive accuracy and generalization performance. Technically, it produced the highest R², lowest RMSE, and strongest cross-validation performance among all tested models. Polynomial expansion allowed the model to capture nonlinear relationships that simple linear models missed, while Ridge regularization controlled coefficient growth and prevented severe overfitting.

From a business perspective, accurate house price prediction can improve property valuation, investment analysis, pricing recommendations, and real estate decision-making. A model with lower prediction error provides more reliable estimates for buyers, sellers, and agencies. The pipeline approach also makes deployment easier because preprocessing, feature transformation, scaling, and prediction are combined into a single reusable workflow.

7. Reflection

One of the hardest concepts in this project was understanding the tradeoff between bias and variance, especially when interpreting learning curves and regularization effects. It was initially surprising to see how dramatically polynomial feature expansion increased the number of features and how quickly this could lead to overfitting. Another interesting insight was observing how Lasso automatically removed weak features while Ridge kept all coefficients but shrank them gradually.

The biggest takeaway from the project was the importance of preprocessing and feature engineering in machine learning performance. Small preprocessing decisions such as log transformation, scaling, and encoding had major impacts on final accuracy. If continuing this project, the next step would be exploring advanced ensemble methods such as Random Forest, XGBoost, and Gradient Boosting to compare their performance against traditional regression models.
