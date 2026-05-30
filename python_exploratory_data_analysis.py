# -----------------------------------
# 1. Import Libraries
# -----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# 2. Load Dataset
# -----------------------------------
file_path = "C:/Users/mahes/OneDrive/Desktop/Projects/customer-shopping-behavior-analysis-dashboard/raw_customer_shopping_data.csv"
df = pd.read_csv(file_path)


# -----------------------------------
# 3. Preview Data
# -----------------------------------
print("========== Top 5 Rows ==========")
print(df.head())

print("\n========== Bottom 5 Rows ==========")
print(df.tail())


# -----------------------------------
# 4. Dataset Overview
# -----------------------------------
print("\n========== Dataset Overview ==========")

# Shape
print("\nShape of Dataset:")
print(df.shape)

# Columns
print("\nColumn Names:")
print(list(df.columns))

# Info
print("\nDataset Info:")
df.info()

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())


# -----------------------------------
# 5. Data Cleaning
# -----------------------------------
print("\n========== Data Cleaning ==========")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unwanted columns if present
drop_cols = ['Unnamed: 0', 'Customer ID']

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

print("\nData Cleaning Completed Successfully ✔️")


# -----------------------------------
# 6. Feature Engineering
# -----------------------------------
print("\n========== Feature Engineering ==========")

# Age Group Column
df['Age Group'] = pd.cut(
    df['Age'],
    bins=[0, 18, 30, 45, 60, 100],
    labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

print("New Feature Added: Age Group")


# -----------------------------------
# 7. Exploratory Data Analysis
# -----------------------------------
print("\n========== EDA ==========")

# Total Customers
total_customers = df.shape[0]

print("\nTotal Customers:")
print(total_customers)

# Average Purchase Amount
avg_purchase = df['Purchase Amount (USD)'].mean()

print("\nAverage Purchase Amount:")
print(avg_purchase)

# Gender Wise Spending
gender_spending = df.groupby('Gender')['Purchase Amount (USD)'].sum()

print("\nGender Wise Spending:")
print(gender_spending)

# Category Wise Sales
category_sales = df.groupby('Category')['Purchase Amount (USD)'].sum()

print("\nCategory Wise Sales:")
print(category_sales)

# Payment Method Analysis
payment_method = df.groupby('Payment Method')['Purchase Amount (USD)'].sum()

print("\nPayment Method Analysis:")
print(payment_method)

# Seasonal Sales
season_sales = df.groupby('Season')['Purchase Amount (USD)'].sum()

print("\nSeason Wise Sales:")
print(season_sales)


# -----------------------------------
# 8. Customer Behavior Analysis
# -----------------------------------
print("\n========== Customer Behavior Analysis ==========")

# Frequency of Purchases
purchase_frequency = df['Frequency of Purchases'].value_counts()

print("\nPurchase Frequency:")
print(purchase_frequency)

# Discount Applied Analysis
discount_analysis = df.groupby('Discount Applied')['Purchase Amount (USD)'].mean()

print("\nDiscount Impact on Purchase Amount:")
print(discount_analysis)

# Review Rating Analysis
review_analysis = df.groupby('Review Rating')['Purchase Amount (USD)'].mean()

print("\nReview Rating vs Purchase Amount:")
print(review_analysis)


# -----------------------------------
# 9. Visualizations
# -----------------------------------

# Gender Wise Spending
plt.figure(figsize=(8,5))

gender_spending.plot(kind='bar')

plt.title("Gender Wise Spending")
plt.xlabel("Gender")
plt.ylabel("Purchase Amount")

plt.show()


# Category Wise Sales
plt.figure(figsize=(10,5))

category_sales.plot(kind='bar')

plt.title("Category Wise Sales")
plt.xlabel("Category")
plt.ylabel("Purchase Amount")

plt.xticks(rotation=45)

plt.show()


# Season Wise Sales
plt.figure(figsize=(8,5))

season_sales.plot(kind='bar')

plt.title("Season Wise Sales")
plt.xlabel("Season")
plt.ylabel("Purchase Amount")

plt.show()


# Payment Method Analysis
plt.figure(figsize=(8,5))

payment_method.plot(kind='bar')

plt.title("Payment Method Analysis")
plt.xlabel("Payment Method")
plt.ylabel("Purchase Amount")

plt.xticks(rotation=45)

plt.show()


# -----------------------------------
# 10. Save Clean Data
# -----------------------------------
print("\n========== Saving Clean Data ==========")

save_path = "C:/Users/mahes/OneDrive/Desktop/Projects/customer-shopping-behavior-analysis-dashboard/cleaned_customer_shopping_data.csv"
df.to_csv(save_path, index=False)

print("Clean Data Saved Successfully ✔️")