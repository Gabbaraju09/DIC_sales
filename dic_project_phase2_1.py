# -*- coding: utf-8 -*-
"""dic_project_phase2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16rjqg07WPwH7_nJLwHn-vWqhmna8gNkJ
"""

#Importing the data
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

url = "/Users/mymac/Downloads/Big Sales Data (1).csv"
data = pd.read_csv(url)
data.head()
print(data)

print("Column Information:")
print(data.info())

"""**Data Cleaning**

**Finding Missing Values**
"""

import numpy as np

data['Item_Weight'].replace('', np.nan, inplace=True)
total_missing = data.isna().sum()
print(total_missing)

# Check specifically for missing values in 'Item_Weight'
missing_item_weight = data['Item_Weight'].isna().sum()
print(f"Missing values in 'Item_Weight': {missing_item_weight}")
# Get rows where 'Item_Weight' is missing (NaN or blank)
missing_data = data[data['Item_Weight'].isna()]
print(missing_data)

"""**1.Handling Missing Values with Mean**"""

# Filling missing values in 'Item_Weight' with the mean of the column
data['Item_Weight']= data['Item_Weight'].fillna(data['Item_Weight'].mean())
print(data['Item_Weight'])

"""The missing values in Item_Weight column are corrected by taking mean value of the existing non-missing weights in that column to fill these values.

**2.Standardize Categorical Data**
"""

# Standardizing 'Item_Fat_Content' column
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})
print(data['Item_Fat_Content'])

"""The column Item_Fat_Content contains the same information in different formats (ex: “LF”, ’low fat” and “Low Fat”). So, converted “LF” and “low fat” with “Low Fat” and “reg” with “Regular”.

**3.Drop Duplicate Rows**
"""

# Drop duplicates and return a new DataFrame
cleaned_data = data.drop_duplicates()
print(cleaned_data)

"""Duplicate rows are dropped

**4.Encoding Categorical Variables**
"""

# One-hot encoding for 'Outlet_Type'
data = pd.get_dummies(data, columns=['Outlet_Type'], drop_first=True)
print(data)

"""converted the categorical Outlet_Type column into multiple columns.

**5.Dropping unnecessary column**
"""

# Check if 'Outlet_Identifier' exists before dropping
if 'Outlet_Identifier' in data.columns:
    data.drop(['Outlet_Identifier'], axis=1, inplace=True)
print(data.head())

"""Dropped Outlet_Identifier column as it is not necessary for our analysis

**6.Creating new row**
"""

data['Outlet_Age'] = 2024 - data['Outlet_Establishment_Year'].astype(int)
print(data['Outlet_Age'])

"""Added a new column named Outlet_Age which computes the age of each outlet depending on the year it was established.

**7.Dropping rows with 0 visibility**
"""

zero_visibility = (data['Item_Visibility'] == 0).sum()
print(f"Number of items with 0 visibility before dropping: {zero_visibility}")
data = data[data['Item_Visibility'] != 0]
zero_visibility_after = (data['Item_Visibility'] == 0).sum()
print(f"Number of items with 0 visibility after dropping: {zero_visibility_after}")

print(data)

"""Rows with Item_Visibility = 0 are dropped

**8.Datetime conversion**
"""

# Convert 'Outlet_Establishment_Year' from int to datetime (year format)
data['Outlet_Establishment_Year'] = pd.to_datetime(data['Outlet_Establishment_Year'], format='%Y')
print(data['Outlet_Establishment_Year'])

"""Converted 'Outlet_Establishment_Year' column from integer (year) to datetime format.

**9.Standardizing values in Outlet_Size**
"""

# Standardizing the values in 'Outlet_Size' column
data['Outlet_Size'] = data['Outlet_Size'].replace({
    'Small': 'Small',
    'Medium': 'Medium',
    'High': 'Large'
})
print(data['Outlet_Size'])

"""The purpose is to standardize the values in the 'Outlet_Size' column such that the categories are consistent and relevant. In this situation, 'High' is replaced with 'Large’, implying that 'High' refers to a large outlet.

**10.Converting to lowercase**
"""

# Convert 'Item_Type' to lowercase
data['Item_Type'] = data['Item_Type'].str.lower()
print(data['Item_Type'])

"""transforms all the strings in the 'Item_Type' column to lowercase

**11. Outlier Detection**
"""

import numpy as np

# Calculate Q1 (25th percentile) and Q3 (75th percentile) for 'Item_Outlet_Sales'
Q1 = data['Item_Outlet_Sales'].quantile(0.25)
Q3 = data['Item_Outlet_Sales'].quantile(0.75)

# Calculate the Interquartile Range (IQR)
IQR = Q3 - Q1

# Define the lower and upper bound to detect outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = data[(data['Item_Outlet_Sales'] < lower_bound) | (data['Item_Outlet_Sales'] > upper_bound)]
print(f"Number of outliers: {outliers.shape[0]}")
print(outliers[['Item_Outlet_Sales']])

"""**Printing Cleaned data**"""

print(data)

"""**Exploratory Data Analysis (EDA)**

**1. Descriptive Statistics**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Descriptive Statistics
print("Descriptive Statistics:")
print(data.describe())

"""provides summary statistics of the numeric columns in the dataset. It returns important measures such as  mean,standard deviation, min and max values etc.

**2.Categorical Features**
"""

# 2. Count plots for categorical features
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Item_Type')
plt.title('Count of Items by Type')
plt.xticks(rotation=45)
plt.show()

# Bar plot for categorical variable 'Outlet_Size' using 'hue'
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Outlet_Size', hue='Outlet_Size', palette='Set2', legend=False)
plt.title('Distribution of Outlet Size')
plt.xlabel('Outlet Size')  # Label for x-axis
plt.ylabel('Count')  # Label for y-axis
plt.show()

"""The count of items by type plot is used to show the frequency (or count) of each unique value in a categorical feature.
The "Distribution of Outlet Size" chart displays the count of outlets across three different size categories:

**3.Univariate Analysis**
"""

# 3. Univariate Analysis
# Histograms for numerical features
data.hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Histograms of Numerical Features')
plt.show()

"""It shows the histogram for the distribution of numerical data columns.

**4.Bivariate Analysis**
"""

# 4. Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Item_Visibility', y='Item_Outlet_Sales')
plt.title('Scatter Plot of Item Visibility vs. Item Outlet Sales')
plt.show()

"""The scatter plot illustrates the relation between Item_Visibility and Item_Outlet_Sales.
The density of dots is higher at the lower end of both axes, indicating that many items have limited visibility and thus low sales statistics.

**5.Feature Distribution Comparison**
"""

# 5. Feature Distribution Comparison
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Item_Type', y='Item_Outlet_Sales')
plt.title('Item Outlet Sales by Item Type')
plt.xticks(rotation=45)
plt.show()

"""Each box represents the distribution of Item_Outlet_Sales for a specific Item_Type.
The center line inside each box shows the item type's median sales.
The box's borders represent the first (Q1) and third (Q3) quartiles, which correspond to the data's(IQR).
The "whiskers" include both the smallest and greatest numbers that are not considered outliers.

**6.Heatmap to visualize the correlations**
"""

# 6. Heatmap to visualize the correlations

# Display data types of each column
print(data.dtypes)

# Select only numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Optionally handle missing values
numeric_data.fillna(numeric_data.mean(), inplace=True)

# Calculate the correlation matrix
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""The highest positive correlation (0.53) exists between Item_MRP and Item_Outlet_Sales, indicating that the more priced items have higher sales.
There is a weak positive association (0.16) between Item_Weight and Item_Outlet_Sales.
Item_Visibility has a negative association (-0.16) with Item_Outlet_Sales, which seems illogical.
Outlet_Age has relatively weak correlations with the majority of other variables, with the largest being a weak positive correlation (0.1) with item_outlet_sales.

**7.Line Plot**
"""

# 7. Line plot to explore the effect of years of operation on sales
plt.figure(figsize=(8,6))
sns.lineplot(x='Outlet_Age', y='Item_Outlet_Sales', data=data)
plt.title('Years of Operation vs Item Outlet Sales')
plt.show()

"""The line plot is used to the link between the number of years an outlet has been open (Outlet_Age) and the associated sales of that outlet (Item_Outlet_Sales).
Sales are generally increasing as outlets age increases.
Sales begin around 2000 units for 15-year-old stores.
There is a very constant period of 20 to 35 years, with sales hovering around 2200 units.
Sales fall sharply to 1000 units after 25 years.

**8.Pairplot**
"""

# 8. Pairplot
sns.pairplot(numeric_data)
plt.title('Pair Plot of Numeric Variables')
plt.show()

"""Pair plots display scatter plots for every combination of numeric variables

**9.Violin Plot**
"""

# 9. Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x='Item_Type', y='Item_Outlet_Sales')
plt.title('Item Outlet Sales Distribution by Item Type')
plt.xticks(rotation=45)
plt.show()

"""This graphic depicts the distribution of item outlet sales across different product kinds.Each violin represents a separate item category, and its breadth denotes the density of data points at various sales values, allowing for a clear visualisation of the distribution's shape, central tendency, and variability.
The y-axis shows item outlet sales, which range from 0 to around 30,000. Each violin plot contains a box plot, denoted by a rectangle (showing the interquartile range) and a line inside.

**10.Pie chart**
"""

# 10. Pie chart
item_type_counts = data['Item_Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(item_type_counts, labels=item_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Item Types')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

"""The chart provides a visual representation of the relative proportions of different item types. The largest categories are fruits and vegetables and snack foods, each making up about 14% of the total. The smallest category is seafood at 0.6%.

**11.KDE plot for Item Outlet Sales**
"""

# 11. KDE plot for Item Outlet Sales
plt.figure(figsize=(10, 6))
sns.kdeplot(data['Item_Outlet_Sales'], fill=True)
plt.title('KDE Plot of Item Outlet Sales')
plt.xlabel('Item Outlet Sales')
plt.ylabel('Density')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv(url)

# Split into features and target variable
data= data.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type'], axis=1)
#data.fillna(data.mean(), inplace=True)

"""**1. Linear Regression**"""

from sklearn.metrics import classification_report, confusion_matrix
X = data[['Item_Outlet_Sales']]
Y = data['Item_MRP']
X_training, X_testing, y_training, y_testing = train_test_split(X, Y, test_size=0.2, random_state=42)

# Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_training, y_training)
y_pred = lin_reg.predict(X_testing)

# Visualization
plt.scatter(X_testing['Item_Outlet_Sales'], y_testing, color='blue')
plt.plot(X_testing['Item_Outlet_Sales'], y_pred, color='red', linewidth=2)
plt.title('MRP vs Sales')
plt.xlabel('Item_Outlet_Sales')
plt.ylabel('Item_MRP')
plt.show()

# Coefficients and performance
print("Coefficients:", linear_reg.coef_)
print("Intercept:", linear_reg.intercept_)
mse = mean_squared_error(y_testing, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print("R² score:", linear_reg.score(X_testing, y_testing))

"""It could be observed from the scatter plot with the regression line that the actual data points had been randomly scattered in a manner such that little linear pattern could be achieved, therefore justifying the low R² score.
Linear Regression, tended to underfit, it captured the general trends but missed more complex variable to variable interactions between them.

**2. K Means**
"""

from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
X_kmeans = data[['Item_Visibility','Item_Outlet_Sales']]

# k-means model with 2 clusters
k_means = KMeans(n_clusters=2, random_state=42)
k_means.fit(X_kmeans)
data['cluster'] = k_means.labels_

# Visualization
plt.scatter(data['Item_Visibility'], data['Item_Outlet_Sales'], c=data['cluster'], cmap='viridis')
plt.title('K-Means Clustering of Products')
plt.xlabel('Item_Visibility')
plt.ylabel('Item_MRP')
plt.show()

# Evaluation Metrics
# 1. Inertia (Sum of Squared Distances to the Closest Cluster Center)
Inertia = k_means.inertia_
print(f"Inertia (Sum of Squared Distances): {Inertia}")

# 2. Silhouette Score (How clusters are separated)
s_score = silhouette_score(X_kmeans, k_means.labels_)
print(f"Silhouette Score: {s_score}")

"""By the clustering analysis, the high visibility products may fall into one group, say low sales, while low visibility ones fall into another.

**3. SVM**
"""

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

X_svm = data[['Item_MRP']]
y_svm = data['Item_Outlet_Sales']
X_training_svm, X_testing_svm, y_training_svm, y_testing_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

# SVR model (RBF kernel for non-linear data)
model_svr = SVR(kernel='rbf')
model_svr.fit(X_training_svm, y_training_svm)
y_pred_svm = model_svr.predict(X_testing_svm)

# Visualization
plt.scatter(y_testing_svm, y_pred_svm, color='blue', alpha=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('SVR: Predicted vs Actual Sales')
plt.show()

# Coefficients and performance
mse = mean_squared_error(y_testing_svm, y_pred_svm)
r2 = r2_score(y_testing_svm, y_pred_svm)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""The points are likely well dispersed from the predicted line, most especially on items that have an extremely high or very low price. This easily suggests that the model did not generalize well, failing in essence to capture the complete complexity of the relationship price and sales share.The SVR model, which slightly outperforms the linear regression, thus showing some nonlinear relationships. However, the high MSE suggested that on its own, Item_MRP had too much noise to predict accurately and thus required more features.

**4. Decision Tree**
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

X_dtree = data[['Item_Visibility']]
y_dtree = data['Item_MRP']
X_training_tree, X_testing_tree, y_training_tree, y_testing_tree = train_test_split(
    X_dtree, y_dtree, test_size=0.2, random_state=42
)

# Decision Tree Regressor model
decision_tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
decision_tree_model.fit(X_train_tree, y_train_tree)

# Visualization
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model, feature_names=['Item_Visibility'], filled=True)
plt.title('Decision Tree for Item MRP Prediction')
plt.show()
y_pred_tree = decision_tree_model.predict(X_test_tree)

# Scatter plot of predicted vs actual values
plt.scatter(y_testing_tree, y_pred_tree)
plt.xlabel('Actual Item MRP')
plt.ylabel('Predicted Item MRP')
plt.title('Decision Tree Regressor: Predicted vs Actual MRP')
plt.show()

# Coefficients and performance
mse = mean_squared_error(y_testing_tree, y_pred_tree)
r2 = r2_score(y_testing_tree, y_pred_tree)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""The tree that is generated provides an idea of exactly how this model splits the data along Item Visibility. The topmost node-the root-includes decisions such that there is a split in the data into different branches based on whether Item Visibility is above or below a threshold.
However, because of the restriction of a maximum depth of 3 for the model, the complexity of the tree may not be enough to capture even the more subtle patterns in the data.

**5. Random Forest**
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


X_random = data[['Item_Visibility', 'Item_MRP']]
y_random = data['Item_Outlet_Sales']
X_training_rf, X_testing_rf, y_training_rf, y_testing_rf = train_test_split(
    X_random, y_random, test_size=0.2, random_state=42
)

#Random Forest Regressor model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_training_rf, y_training_rf)
importances = random_forest_model.feature_importances_
plt.barh(['Item_Visibility', 'Item_MRP'], importances)
plt.title('Feature Importance in Random Forest Regressor')
plt.show()
y_pred_rf = random_forest_model.predict(X_testing_rf)

# Scatter plot of predicted vs actual values
plt.scatter(y_testing_rf, y_pred_rf)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Random Forest Regressor: Predicted vs Actual Sales')
plt.show()

# Coefficients and performance
mse = mean_squared_error(y_testing_rf, y_pred_rf)
r2 = r2_score(y_testing_rf, y_pred_rf)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""The bar plot showed the importance of the features:
Item MRP had a much higher importance score than the Item Visibility, thus showing that the price of an item is more strongly correlated with sales than its visibility.Actual vs Predicted Sales Scatter Plot: A scatter plot was created to overlay actual Item Outlet Sales values with the predicted values provided by the Random Forest model.
Random Forest averages many models, it resulted in better performance than a single decision tree. It reduced the variance to stabilize the model and helped in generalization.Importance of Features Analysis Both Item_Visibility and Item_MRP were found contributing toward predicting sales.

**6. Logistic Regresion**
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

data['MRP_Category'] = pd.cut(data['Item_MRP'], bins=[0, 100, 200, 500], labels=[0, 1, 2])

X_logistic_reg = data[['Item_Outlet_Sales']]
y_logistic_reg = data['MRP_Category']
X_training_logreg, X_testing_logreg, y_training_logreg, y_testing_logreg = train_test_split(
    X_logistic_reg, y_logistic_reg, test_size=0.2, random_state=42
)

#Logistic Regression model
logistic_reg_model = LogisticRegression(multi_class='ovr', max_iter=1000)
logistic_reg_model.fit(X_training_logreg, y_training_logreg)
y_pred_logreg = logistic_reg_model.predict(X_testing_logreg)

# ROC Curve
y_prob = logistic_reg_model.predict_proba(X_testing_logreg)
fpr, tpr, _ = roc_curve(y_testing_logreg, y_prob[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

# Visualization
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_testing_logreg, y_testing_logreg, color='blue', label='Actual Categories', alpha=0.5)
plt.scatter(X_testing_logreg, y_pred, color='red', label='Predicted Categories', alpha=0.5)
plt.title('Logistic Regression: Actual vs Predicted MRP Categories')
plt.xlabel('Item Outlet Sales')
plt.ylabel('MRP Categories (0: Low, 1: Medium, 2: High)')
plt.legend()
plt.show()

# Evaluate the model
print("Accuracy:", accuracy_score(y_testing_logreg, y_pred_logreg))

"""ROC-AUC visualizes the trade-off between true positive rate vs. false positive rate for one of the classes (likely medium MRP) and gives an idea of the overall performances to distinguish the medium category from the other categories. In this case, ROC likely gave a moderate performance, reflected by the AUC score of 0.62.
Scatter Plot: The blue dots plot the true categories of Item MRP against each value of Item Outlet Sales.
Red dots indicate the predicted categories of Item MRP.
Transparency (alpha=0.5) is added to help distinguish overlapping points.
Logistic regression reveals that sales are related to the price of an item in terms of price classes. The results concluded that classifying with a single feature of Item_Outlet_Sales was quite constrained and required more features or perhaps other kinds of classification algorithms for good performance.

**7. KNN**
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X_knn = data[['Item_Visibility']]
y_knn = data['Item_MRP']
X_training_knn, X_testing_knn, y_training_knn, y_testing_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=42)

# KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_training_knn, y_training_knn)
y_pred_knn = knn_model.predict(X_testing_knn)

# Scatter Plot: Actual vs Predicted MRP
plt.figure(figsize=(10, 6))
plt.scatter(y_testing_knn, y_pred_knn, color='blue', alpha=0.6)
plt.plot([y_testing_knn.min(), y_testing_knn.max()], [y_testing_knn.min(), y_testing_knn.max()], 'r--')  # Line y=x
plt.xlabel('Actual Item MRP')
plt.ylabel('Predicted Item MRP')
plt.title('KNN Regressor: Actual vs Predicted MRP')
plt.show()

# Residual Plot: Residuals vs Predicted MRP
residuals = y_testing_knn - y_pred_knn
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_knn, residuals, color='green', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Item MRP')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('KNN Regressor: Residual Plot')
plt.show()

# Coefficients and performance
mse = mean_squared_error(y_testing_knn, y_pred_knn)
r2 = r2_score(y_testing_knn, y_pred_knn)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

"""Scatter Plot Actual versus Predicted MRP: A scatter plot to compare the actual values of Item MRP to the predicted values probably was created using the KNN model. In this case, the points would be considerably scattered away from the line, insinuating major errors in the predictions.
Residual Plot (Predicted versus Residuals): The Residual Plot displays the residuals plotted vs. the predicted values. In this case, residuals would show a distinct pattern, indicating that the KNN model has failed to catch the underlying relationship between Item Visibility and Item MRP.
The reason why KNN performed poorly could be that Item Visibility alone is a very poor predictor of Item MRP.
"""