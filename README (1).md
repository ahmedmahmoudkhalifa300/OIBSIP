# Exploratory Data Analysis (EDA) on Retail Sales Data

Welcome to the Retail Sales Data Analysis project! This project explores retail transactions to understand customer demographics, identify sales trends over time, compare product categories, and deliver actionable business recommendations based on statistical findings.

## 1. Project Overview & Key Performance Indicators (KPIs)
Based on the analysis of 1,000 retail transactions, here are our major findings:
- **Total Revenue:** $456,000.00
- **Total Transactions:** 1,000
- **Total Units Sold:** 2,514 items
- **Average Transaction Value:** $456.00
- **Top Performing Product Category:** Electronics ($156,905.00)
- **Peak Sales Month:** May 2023 ($53,150.00)
- **Dominant Customer Gender by Spend:** Female ($232,840.00)
- **Average Customer Age:** 41.39 years old

---

## 2. Key Visualizations & Findings

### A. Monthly Sales Trend
There is a notable fluctuation in monthly sales throughout the year. May 2023 represents the absolute peak of customer activity.

![Monthly Sales Trend](outputs/monthly_sales.png)

### B. Sales Performance by Product Category
Product category performance is remarkably balanced, though Electronics slightly leads, closely followed by Clothing and Beauty products.

![Sales by Product Category](outputs/sales_by_category.png)

### C. Correlation Heatmap
The correlation analysis shows a strong linear relationship (0.85) between `Price per Unit` and `Total Amount`, while other demographic factors show almost no correlation.

![Correlation Heatmap](outputs/correlation_heatmap.png)

---

## 3. Key Insights
1. **Product Balance:** All three categories (Electronics, Clothing, Beauty) perform very closely in revenue, suggesting steady multi-category demand.
2. **Gender Spend Pattern:** Female shoppers account for slightly more transactions and revenue ($232,840.00) compared to male shoppers ($223,160.00).
3. **Age & Spend Independence:** Customer age holds a very weak correlation (-0.06) with purchase amount, meaning marketing campaigns shouldn't restrict their targeting by age brackets.

## 4. Business Recommendations
1. **Seasonal Campaigns:** Design and run high-budget marketing campaigns in early Q2 to fully capitalize on the natural seasonal sales peak in May.
2. **Product Bundling:** Create promotional bundles that combine elements of Beauty/Clothing with high-ticket Electronics to increase the average order quantity per transaction.
3. **Inclusive Customer Targeting:** Focus advertising campaigns on interests and category affinity rather than strict age/gender demographics, as purchasing patterns are uniform across various demographics.
