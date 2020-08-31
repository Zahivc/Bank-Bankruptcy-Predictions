# Bank-Bankruptcy-Predictions
Authors: Jeff Cheah Vyi, Yuyao Chen, Harry Tsang Tsz Hin, Soh Harn Yi Royston, Li Yanning


## 1. Introduction

The collapse of Lehman Brothers in 2008 caused shockwaves worldwide, resulting in the Dow Jones to have a single-day 4.4% drop, the biggest plunge since the September 11 attacks. It also resulted in huge losses across money funds and hedge funds, and resulted in many investors losing their money, and for some investors, losing their life savings. It was devasting for millions of people across the globe, and our group intends to build a model to predict such bank failures and pre-empt any investors and help them make more prudent investments.

To do this, our group will be focusing on the banks in the United States of America (USA), the largest financial market globally. USA has set up an independent federal agency, the Federal Deposit Insurance Corporation (FDIC) in 1933, to maintain stability and public’s confidence in the country’s financial system. The FDIC provides deposit insurance for US-based commercial banks and charges them insurance premium according to an internal (non-public) rating based on the CAMELS supervisory rating system. Whenever a FDIC-insured bank fails, FDIC ensures the bank’s customers has prompt access to their insured deposits. A bank is considered is considered to have failed when it is closed by the federal or state banking regulatory, generally due to the bank being unable to meet its obligations to depositors or others.

## 2. Business Tasks 

Our group aims to build a Classification Predictor, to predict if a bank likely to fail the next quarter based on 1-year financial ratios and other macro-economical attributes. Our model will also interpret the classifier and tell the user which factors contribute most to the predicted results. 

We believe this model can be used as an early warning system for regulators and the investors, which would help regulators in their policy making and help investors making the best choices of whether or not to invest in a bank stock or to invests in funds with huge exposures to the bank stocks. Our model can also be used by banks to decide the creditworthiness of another bank to decide the interbank lending rate. Additionally, our models can be used by banks to improve the bank’s own internal rating models.

## 3. Data Sources

The FDIC monitors the financial institution for safety and soundness by monitoring several performance ratios and they upload this data to their website. Our team has downloaded FDIC Bank Failure Public Dataset which includes the quarterly financial data of individual banks and the bank failure labels. Our group has also collected macroeconomic data across the same time period and includes factors such as term spread, stock market growth, and real Gross Domestic Product (GDP) growth amongst others. 

Our dataset is a collection of quarterly financial data of individual banks between the year of 2001 to 2015, which includes their financial information from, assets, loans, earnings and liquidity. The main y variable here is whether the bank failed in that quarter, which is binary. We also supplement this data with external macroeconomic factors, such as term spread, stock market growth and real GDP growth to reflect the prevailing market conditions.

## 4. Preprocessing

a. After calculation, we have the following features: 
1) Quarterly Financial Data: 
Total Assets (TA), Net Income to TA, Equity to TA, ROA, Core Deposits to TA
Non-performing Loan (NPL) to Total Loan (TL), Loss Provision to_TL, Allowance for Loan and Lease Losses (ALLL) to TL

2) Quarterly Macro Data:
Term Spread, Stock Mkt Growth, Real Gdp Growth, Unemployment Rate Change, Treasury Yield 3M, BBB Spread.

b. Split the banks by the time of the bank failure. The banks that failed before 2011 will belong to the train dataset; the banks that failed between 2011 and 2012 will belong to the validation dataset; the banks that failed after 2012 will belong to the test dataset. 

As a result, we have 326 banks in train, 66 banks in validation and 70 banks in test. 


![GitHub Logo](/images/prep1.png)

Extract the data according to Bank ID Index for Train/Validation/Test datasets. 

![GitHub Logo](/images/prep2.png)

c. For each bank, organize the features and the label. Each data record has 1-year financial indicators as features and the label, which is whether the bank will fail in the next quarter.

![GitHub Logo](/images/prep3.png)

![GitHub Logo](/images/prep4.png)

![GitHub Logo](/images/prep5.png)

d. As a result, we will finally get our final train, validation and test dataset. 

![GitHub Logo](/images/prep6.png)

## 5. Modelling

### 5.1 Machine Learning

The below tables show the general steps for machine learning approaches:

![GitHub Logo](/images/ml0.png)

#### 5.1.1 Feature Engineering

##### A.	Log / Scaling
After reviewing the scaling of our features, we found that “Max_total_asset” was much larger than other features. Thus, we firstly used “log1p” to compute the log (Max_total_asset + 1). 
We also tried the scaling technique, such as MinMaxScaler. However, the results showed that the unscaled dataset performed better than the scaled ones. 

#### B.	Feature Generation: Compute the Quarterly Change
In machine learning, the models we used could not take the time series relationship into consideration. Thus, we aimed to add some time series information by computing the quarterly change of each metric. The function “feature_generate” was defined as below: 

![GitHub Logo](/images/ml1.png)

##### Figure 1: Feature Engineering

Take the metric “NI_To_TA” for example. After preprocessing, each data point consists of 4 quarters’ records, including “NI_To_TA_1”, “NI_To_TA_2”, “NI_To_TA_3” and “NI_To_TA_4”. Thus, we computed the quarterly change from 1st quarter to 2nd quarter, from 2nd quarter to 3rd quarter, from 3rd quarter to 4th quarter. 

#### 5.1.2 Machine Learning Modelling

In the modelling part, we choose 4 methods in total, including Logistic Regression, Random Forest, Gradient Boosting and Stacking. Logistic Regression serves as a baseline model in this project. Random Forest, Gradient Boosting and Stacking represent three important types of ensemble models. 

#### A.	Logistic Regression
1)	Hyper-parameter Tuning: The only hyperparameter we tuned for logistic regression is “C”, which represents the inverse of regularization strength. We used GridSearchCV to do the cross validation and search for the best parameter. The scoring metric we used in GridSearchCV is “roc_auc” because we believe ROC AUC score serves as a good metric for model evaluation for imbalanced dataset. 

![GitHub Logo](/images/ml2.png)

##### Figure 2: Hyper-parameter Tuning

The result from GridSearchCV showed that the best parameter for Logistic Regression is C = 0.1 when ROC_AUC Score is 0.95. 
2)	Model Training Using Best Parameter: After we had the best parameter, we re-trained the model with the best parameter (C = 0.1) using the whole train dataset. 

3)	Threshold Tuning: We wanted to choose a threshold that could help us emphasize more on recall than precision. Thus, we defined a function called “print_report”. Taking the Y_True and the pred_proba value of positive class of the model as inputs, the function could output the Precision, Recall, F2 Score based on various thresholds. Finally, we decided to use i = 0.3 for Logistic Regression Model. 
 
![GitHub Logo](/images/ml3.png)

##### Figure 3: Threshold Tuning
4)	Feature Importance: The below bar chart showed the top features with greatest absolute coefficients. The result showed that the general accounting metrics, such as Equity to TA, Max Total Asset, contribute most to the model prediction. Besides, macro-economic indicators, such as 3-month Treasury Yield, unemployment rate change, and BBB Spread, influence the prediction greatly.

![GitHub Logo](/images/ml4.png)
 
##### Figure 4: Feature Importance (Log Regression)
#### B.	Random Forest 
1) Hyper-parameter Tuning: To tune the hyper-parameter for Random Forest, we used RandomizedSearchCV because it is generally quicker compared with GridSearchCV if we want to try various combinations of hyper-parameters. In terms of parameter distributions, we consider the following:
•	“n_estimators”: The parameter specifies the number of trees used in the forest of the model. Here we tried 100, 325, 550, 775, 1000. 
•	“max_features”: The parameter specifies the number of features to consider. “auto” means sqrt (n_features). 
•	“max_depth”: The maximum depth of a tree. Here we considered 10, 32, 55, 77, 100 and None. If none, the nodes will be expanded until all leaves are pure.  
•	“min_samples_split”: The min_samples_split specifies the minimum number of samples required to split the leaf node. We considered 2, 5, 10 here. 
•	“min_samples_leaf”: The min_samples_leaf specifies the minimum number of samples required to constitute a leaf node. We considered 1, 2, 4 here. 
 
![GitHub Logo](/images/ml5.png)
![GitHub Logo](/images/ml6.png) 

##### Figure 5: Random Forest
According to the result of RandomizedSearchCV, the best parameters were shown as below:
 
![GitHub Logo](/images/ml7.png) 

##### Figure 6: Best Parameters
After we had the best parameters, we re-trained the Random Forest model and tuned the threshold. The threshold we chose was i = 0.3. 
2) Feature Importance: The below bar chart showed the most important features that contributed to the Random Forest model. Similar to Logistic Regression model, the most significant feature in Random Forest model was still Equity to TA; also, metrics relating to 3M Treasury Yield were very important in the Random Forest model. Still, Random Forest model seemed to value the loan-related metrics more compared with Logistic Regression model since NPL_To_TL and ALLL_To_TL contributed greatly to the prediction. 

![GitHub Logo](/images/ml8.png)  

##### Figure 7: Feature Importance: Random Forest
#### C.	Gradient Boosting
1) Hyper-parameter Tuning: We didn’t use GridSearchCV or RandomizedSearchCV when tuning hyper-parameters for the Gradient Boosting model because the two exhaustive methods were very time-consuming. Here we manually tuned two hyper-parameters, “n_estimators” and “learning_rate”. After tuning we decided to use n_estimators = 50 and learning_rate = 0.1. The threshold we used for Gradient Boosting was 0.2. 

##### ![GitHub Logo](/images/ml9.png) 

2) Feature Importance: The top features of Gradient Boosting model were similar to Random Forest model. Still, in Gradient Boosting model, the feature “Equity_To_TA_Q4” dominated among the features.  

![GitHub Logo](/images/ml10.png) 

##### Figure 8: Feature Importance: Gradient Boosting
#### D.	Stacking Model 
In Stacking model, we used Random Forest and Logistic Regression as our base classifiers and used Logistic Regression as our final classifier. 
Similar to Gradient Boosting, it is very time-consuming to use exhaustive methods to tune the hyper-parameters. Therefore, we also tuned the hyper-parameters manually. The main hyper-parameter we tried was the “n_estimators” of the base model Random Forest. Eventually we chose n_estimators = 200. The threshold we used was i = 0.2. 

![GitHub Logo](/images/ml11.png) 

#### 5.1.3. Model Results and Evaluations 

After we got the models, we applied the four models to the test dataset. The results were reported below:
Evaluation Metric	Logistic Regression	Random Forest	Gradient Boosting	Stacking

![GitHub Logo](/images/model_comparison.png) 

##### Table 1: Evaluation Metrics

In terms of ROC AUC Score, Random Forest and Stacking perform slightly better than the other two models. If we look at F2 score, which provides a more balanced view of Recall and Precision, stacking model performs best. In general, all the models are better at Recall than Precision, which means the models are better at capturing the true positives but may have many false positives.

Although the machine learning approach already gave us a fairly good result, there are some limitations:
1. Limited Information on Times Series Change: Ideally, we would want the models to consider the times series relationship. We tried to reflect the quarterly change in the feature engineering part, but it was still very limited.
2. Too Many Hand-crafted Features: Feature Engineering is an important step in machine learning modelling, but it is very time-consuming and not very efficient. In the exploration stage, we tried various ways of feature generation. However, the results showed the newly generated features could contribute limited improvement to model performance. 
Given the limitations of machine learning approaches, our second approach was deep learning, using LSTM to uncover more relationship that could not be considered in machine learning approaches.  


### 5.2. LSTM

LSTM model provides the ability to deal with time sequence data by its feedback connections. In order to feed the data into LSTM model. We need to reshape the data into three dimensions. In our dataset, there are four timesteps each with 22 features. Therefore, several data preprocessing steps are required. The details are elaborated as below.

#### 5.2.1. Detailed Steps 

##### a. Oversampling

There is always an imbalanced issue in corporate bankruptcy dataset. Fortunately, it is proved that the neural network performs well in bankruptcy prediction with equal numbers of examples in the learning phase [1]. Therefore, the oversampling algorithm - SMOTE was employed to tackle the data imbalance issue.

##### b. Reshape and Normalization

Reshape the train, validation and test data to 3 demensions – (Observation, Timestep, Feature) to satisfy the model requirements, convert the response variable y to integer and normalize the predictor variable x by MinMaxScaler.

#### c. Model Structure
 
1.	4 LSTM layers, with 128, 64, 32, 16 neurons.
2.	Use “relu” as the activation function.
3.	Dense layer with 1 neuron as the output of the model  
4.	Dropout rate = 0.5 to avoid overfitting.
5.	Set up the epoch to 4 to avoid overfitting.





