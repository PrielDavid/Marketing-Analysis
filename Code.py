# Marketing Analysis with Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from pathlib import Path

# ---------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------
DATA_PATH = Path('C:\projects\Marketing Analysis\data\Sales Transaction v.4a.csv')
EXPORT_DIR = Path('exports')
FIG_DIR = Path('figures')
EXPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Price'] * df['Quantity']

# Remove negative or zero quantity/price if they exist
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

# ---------------------------------------------------------------------
# 2. BUSINESS QUESTION 1 — Product Performance
# ---------------------------------------------------------------------
product_rev = (df.groupby('ProductName')['Revenue']
                 .sum()
                 .sort_values(ascending=False)
                 .reset_index())

# Save for Power BI
EXPORT_DIR.mkdir(exist_ok=True)
product_rev.to_csv(EXPORT_DIR / 'product_revenue.csv', index=False)

# Top‑15 bar plot
top15 = product_rev.head(15)
plt.figure(figsize=(10, 6))
sns.barplot(y='ProductName', x='Revenue', data=top15, orient='h')
plt.title('Top 15 Products by Total Revenue')
plt.tight_layout()
plt.savefig(FIG_DIR / 'top15_products.png')

# Monthly trend for top 3 products
top3_products = top15['ProductName'][:3].tolist()
monthly_prod = (df[df['ProductName'].isin(top3_products)]
                .groupby([pd.Grouper(key='Date', freq='M'), 'ProductName'])
                ['Revenue'].sum().reset_index())

fig, ax = plt.subplots(figsize=(10, 6))
for name, grp in monthly_prod.groupby('ProductName'):
    ax.plot(grp['Date'], grp['Revenue'], marker='o', label=name)
ax.set_title('Monthly Revenue Trend – Top 3 Products')
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / 'monthly_top3_trend.png')

# ---------------------------------------------------------------------
# 3. BUSINESS QUESTION 2 — Country Performance
# ---------------------------------------------------------------------
country_metrics = (df.groupby('Country')
                     .agg(Revenue=('Revenue', 'sum'),
                          Orders=('TransactionNo', 'nunique'),
                          Items=('Quantity', 'sum'))
                     .reset_index())
country_metrics['Avg_Order_Value'] = country_metrics['Revenue'] / country_metrics['Orders']
country_metrics = country_metrics.sort_values('Revenue', ascending=False)

# Monthly revenue per country (top 5)
top_countries = country_metrics.head(5)['Country'].tolist()
country_monthly = (df[df['Country'].isin(top_countries)]
                   .groupby([pd.Grouper(key='Date', freq='M'), 'Country'])
                   ['Revenue'].sum().reset_index())
country_monthly.to_csv(EXPORT_DIR / 'country_monthly_revenue.csv', index=False)

plt.figure(figsize=(10, 6))
for country, grp in country_monthly.groupby('Country'):
    plt.plot(grp['Date'], grp['Revenue'], marker='o', label=country)
plt.title('Monthly Revenue Trend – Top Countries')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'monthly_country_trend.png')

# ---------------------------------------------------------------------
# 4. BUSINESS QUESTION 3 — Customer Segmentation (RFM + KMeans)
# ---------------------------------------------------------------------
snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
rfm = (df.groupby('CustomerNo').agg(
        Recency=('Date', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionNo', 'nunique'),
        Monetary=('Revenue', 'sum'))
       .dropna())

# Log transform to reduce skew
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Save segments
rfm.reset_index().to_csv(EXPORT_DIR / 'customer_segments.csv', index=False)

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Cluster', palette='tab10')
plt.title('Customer Segments – Recency vs Monetary')
plt.tight_layout()
plt.savefig(FIG_DIR / 'rfm_clusters.png')

# ---------------------------------------------------------------------
# 5. BUSINESS QUESTION 4 — Revenue Forecast (Prophet)
# ---------------------------------------------------------------------
daily_rev = df.groupby(pd.Grouper(key='Date', freq='D'))['Revenue'].sum().reset_index()
daily_rev.columns = ['ds', 'y']
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model.fit(daily_rev)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
fig1.savefig(FIG_DIR / 'revenue_forecast.png')

# Save forecast for Power BI
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
    EXPORT_DIR / 'daily_revenue_forecast.csv', index=False)

# ---------------------------------------------------------------------
# 6. (Optional) Load to SQLite for Advanced SQL
# ---------------------------------------------------------------------
engine = create_engine('sqlite:///sales.db', echo=False)
df.to_sql('sales', con=engine, if_exists='replace', index=False)

# Example SQL query:
query = """
SELECT
    strftime('%Y-%m', Date) AS YearMonth,
    Country,
    SUM(Price * Quantity) AS Revenue
FROM sales
GROUP BY YearMonth, Country
ORDER BY Revenue DESC
LIMIT 10;
"""
print(pd.read_sql_query(query, engine).head())


