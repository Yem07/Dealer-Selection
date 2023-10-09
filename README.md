# MLM

# Project Readme: Trade Execution Optimization

## Overview
This project focuses on optimizing trade executions in the financial domain, particularly in the context of dealer-client trading pairs. The primary objective is to enhance efficiency, reduce time to quote prices, and increase screen-saving amounts per dealer, ultimately leading to improved revenue generation.

## Data Loading and Exploration
The project begins with loading trade data from the 'invesco_complete.xlsx' file using the Pandas library. Initial data exploration involves examining the columns and displaying a summary of the trades dataset.

```python
# Load the data
trades = pd.read_excel(r'invesco_complete.xlsx', engine='openpyxl')

# Display columns and dataset
trades.columns
trades
```

## Data Analysis
Various analyses are performed to understand the dataset and identify key insights. The following code snippets demonstrate grouping trades by dealers and aggregating metrics such as the number of trades, mean screen-saving amounts, and mean accepted volume.

```python
# Group by Dealer and aggregate metrics
trades.groupby('Dealer').agg({'ISIN': 'count', 'Screen saving (price)': 'mean', 'Accepted Vol': 'mean'})
```

## Machine Learning Model for Prediction
The core of the project involves designing and implementing a predictive machine learning model to optimize trade executions. The model focuses on predicting screen-saving amounts based on features such as time to quote and accepted volume.

The following steps are taken:

1. Data Preprocessing:
   - Drop rows with missing values.
   - Define feature columns.

```python
# Drop rows with missing values
trades.dropna(subset=['Dealer', 'Screen Saving (amount)'], inplace=True)

# Define feature columns
feature_columns = ['Time To Quote', 'Accepted Vol']
```

2. Model Training and Prediction:
   - Iterate over unique dealers and train separate models.
   - Use an XGBoost regressor for amount prediction.

```python
# Iterate over unique dealers and train models
for dealer in trades['Dealer'].unique():
   # ... (Code for training and predicting for each dealer)
```

3. Rank Dealers Based on Cheapest Amounts:
   - Rank dealers based on the cheapest screen-saving amounts predicted by the models.

```python
# Rank dealers based on cheapest amounts
ranked_dealers = sorted(dealer_cheapest_amounts, key=lambda x: dealer_cheapest_amounts[x])

# Print ranked dealers
for rank, dealer in enumerate(ranked_dealers, start=1):
   print(f"Rank {rank}: Dealer '{dealer}' with cheapest amount of {dealer_cheapest_amounts[dealer]}")
```

## Model Interpretability and Visualization
The project includes efforts to interpret and visualize the model's behavior. Key components include:

- Permutation Importance:
  - Calculate and display permutation importance scores for each feature.

```python
# Calculate permutation importance
perm_importance = permutation_importance(xgb_regressor, X_numerical, y_amount, n_repeats=30, random_state=42)

# Print permutation importance scores
for i, feature in enumerate(feature_columns):
   print(f'{feature} importance: {perm_importance.importances_mean[i]}')
```

- Partial Dependence Plots:
  - Create and display partial dependence plots for selected features.

```python
# Create and display partial dependence plots
plot_partial_dependence(xgb_regressor, X_train_numerical, features=[1], grid_resolution=50)  # 1 corresponds to "Accepted Vol"
plt.show()
```

## Data Visualization
The project incorporates data visualization techniques using libraries such as Seaborn and Matplotlib. Examples include count plots, scatter plots, and pie charts to illustrate trade-related patterns and distributions.

```python
# Create a countplot using Seaborn
plt.figure(figsize=(10, 6))
sns.countplot(x='Dealer', data=trades, order=trades['Dealer'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Dealer')
plt.ylabel('Number of Trades')
plt.title('Number of Trades per Dealer')
plt.tight_layout()

# Show the plot
plt.show()
```

## Conclusion
This project aims to provide a comprehensive approach to trade execution optimization, from data exploration and analysis to machine learning model development and interpretation. The insights gained and models created can serve as a foundation for continuous improvement in financial trade execution processes.
