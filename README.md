# Customer Lifetime Values
## Context
**A. Business Problem**:

A car insurance company aims to increase their revenue from the sales of car insurance policies by improving customer lifetime value (CLV). To achieve this, they need to predict CLV using the characteristics of their prospective customers based on the company's historical data. Additionally, they want to identify the factors that significantly affect CLV to make informed business decisions. They require a machine learning model that can accurately predict CLV and determine the factors that impact it.

The profit of an insurance company is the difference between the revenue collected from premiums and the expenses associated with paying out claims and running the business. So, the formula for profit would be:

**Profit = Total revenue - Total expenses**

The total revenue can be estimated as the sum of all premiums collected from customers over their lifetime, which can be calculated as:

**Lifetime revenue = Monthly premium auto * Customer lifetime value**

And the total expenses can include claims payouts, operating expenses, and other costs. So, the profit of an insurance company would be the difference between the lifetime revenue and total expenses:

**Profit = Lifetime revenue - Total expenses**

Meaning that when CLV is high, the lifetime revenue per customer to car insurance company is also higher. Then it will make the lifetime revenue of the company is also higher which means that the profit is also getting higher. The impact of CLV is good to raise the profit.

It's important to note that this is a simplified formula and doesn't take into account many factors that can affect an insurance company's profitability.

**B. Stakeholders**:

The car insurance company.

**C. Problem**:

* The car insurance company wants to increase its profits from the sales of car insurance policies.
* Identifying the factors that significantly affect CLV is difficult, making it challenging to make informed business decisions.

**D. Importance of the Problem**:

* Increasing CLV can help car insurance companies generate more revenue and reduce customer acquisition costs.
* Knowing what factors influence CLV can help companies make better-informed and effective business decisions.

**E. Goal**:

* To develop a machine learning model that can accurately predict CLV for each customer, thereby improving the car insurance company's revenue from the sales of car insurance policies. 
* Additionally, the model should identify the factors that impact CLV, which can help the company make informed business decisions.

**F. Analytic Approach**:

The analytical approach taken is to analyze the data in order to discover patterns from the existing features. Implementation of supervised machine learning and regression models will be performed as a tool to predict CLV and identify the factors that influence it. The creation of Machine Learning model, we will select the best model from several Machine Learning Models by measuring the performance error using Mean Absolute Percentage Error(MAPE). We will select the best model with the lowest (near 0) MAPE score evaluation. Then we will tune the model using hyperparameter to improve the performance by decreasing MAPE score. Lastly, we can use this best model to predict the CLV and identify the features importances. 

## Model
### Benchmark Model

The models that we want to try are:
* linear regression,    
* knnregressor,  
* RandomForestRegressor,  
* SVRegressor,  
* regularized linear model,
* xgb regressor

### Metrics
We use MAPE for the metrics. MAPE stands for Mean Absolute Percentage Error. It is a metric used to evaluate the accuracy of a forecasting model by calculating the percentage difference between the predicted and actual values.

To calculate MAPE, the absolute error for each data point is first calculated by taking the absolute difference between the predicted value and the actual value. This absolute error is then divided by the actual value to obtain the percentage error. The average of these percentage errors is then calculated to obtain the MAPE.  

It is useful because it provides a relative measure of the accuracy of the forecast, meaning it can be used to compare the performance of different models or forecasting methods regardless of the scale of the data.  

MAPE is also more interpretable than other metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) because it is expressed as a percentage. This means that a MAPE of, say, 5% indicates that the model is on average off by 5% of the actual value. This can be easier for stakeholders to understand and interpret compared to a metric like RMSE, which is expressed in the same units as the data being forecasted.

### Selected Model
From the model above, we take the best model which is Random Forest with the lowest Mean Absolute Percentage Error, around 10.9% from the actual value.
We use Random forest regressor which is machine learning algorithm used for regression problems. It is an ensemble method that builds multiple decision trees using bootstrap samples of the training data and random subsets of the features. The final prediction is made by aggregating the predictions of all the decision trees.  

Each decision tree is trained using a subset of the training data and a random subset of the features. This helps to reduce overfitting and improves the generalization ability of the model. The number of decision trees to be used in the model is a hyperparameter that can be tuned based on the performance on the validation set.  

The main advantage of using a random forest regressor is that it can handle a large number of input features and can automatically select the most important features for making the predictions. 

### Hyperparameter

## Conclusion
Our analysis involved several key steps in preparing the data and developing the machine learning model. Firstly, we performed data preprocessing to prepare the data for analysis. We transformed the continuous numerical variables using Robust Scaler to minimize the influence of outliers, while the categorical variables were encoded using One Hot Encoding and Binary Encoding to convert them into numerical variables that the model could interpret.

Next, we developed a benchmark model to compare the performance of several models and find the best fit for our data. We evaluated the models based on the Mean Absolute Percentage Error (MAPE) metric and found that the Random Forest Regressor model performed the best with a **MAPE score of 0.109**.

After selecting the Random Forest Regressor model, we further optimized the model by tuning its hyperparameters. This involved adjusting various model parameters, such as the number of trees and the maximum depth of each tree, to improve the model's performance. This tuning process resulted in a slight improvement in the model's performance, with an increased score of 0.1%.

Lastly, we conducted residual analysis to evaluate the performance of the model and identify any limitations. The residual analysis revealed that the model had difficulty predicting high CLV values, which we noted previously. However, we also found that the model performed well overall, with a prediction error of only 12% compared to the actual values.

In conclusion, our machine learning model provides a valuable tool for predicting customer lifetime value and identifying potential high-value customers. We recommend further refinement of the model by incorporating more relevant information especially **Number of Policies and Monthly Premium Auto Values** as those features are more **important than other features** then we can improve its ability to predict high CLV values. By continuing to refine and improve the model, we can help drive business success and increase profitability for the company and more focus on the desired features.

## Recommendation
Based on the conclusion, there are several recommendations that can be made to improve the machine learning model and its application:

1. Incorporate more relevant information: The model could be improved by incorporating additional features, particularly the Number of Policies and Monthly Premium Auto values, which were found to be important predictors of CLV.

2. Obtain more data on high CLV customers: The model had difficulty predicting high CLV values, so obtaining more data on these customers would improve the model's accuracy.

3. Address the overfitting issue: The model showed a slight overfitting issue, which could be addressed by adjusting the model's parameters or using a different algorithm.

4. Refine the model through further tuning: The model could be further refined by tuning its hyperparameters and evaluating its performance with residual analysis.

5. Utilize the model to identify potential high-value customers: By using the model to predict CLV, the company can identify potential high-value customers and take proactive measures to retain them, such as offering personalized promotions or incentives.

6. Monitor feature importance to anticipate changes in CLV: As the market and customer behavior evolve over time, certain features may become more or less important predictors of CLV. By monitoring the feature importance of the model and conducting regular updates, the company can anticipate changes in CLV and adjust their strategy accordingly. For example, if the model shows that customer tenure becomes a more important predictor of CLV, the company can adjust their retention strategy to focus more on retaining long-term customers. Similarly, if the model shows that certain demographics or product features become less important predictors of CLV, the company can adjust their marketing strategy to focus on other areas. By staying proactive in monitoring and updating the model's feature importance, the company can ensure that their CLV predictions remain accurate and effective in driving business decisions.

Overall, the Random Forest model provides a valuable tool for improving business decisions and driving profitability for the company. By implementing the recommendations above and continuing to refine the model, the company can optimize its use and achieve even greater success.
