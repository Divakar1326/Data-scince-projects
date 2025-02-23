# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/Users/diva1/OneDrive/Desktop/task1/Unemployment in India.csv")

# Step 2: Data Cleaning & Preprocessing
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # Standardizing column names
df.rename(columns={'estimated_unemployment_rate_(%)': 'unemployment_rate',
                   'estimated_labour_participation_rate_(%)': 'labour_participation_rate'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])  # Convert date column to datetime

# Step 3: Exploratory Data Analysis (EDA)
print("\nDataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nExample Data")
print(df.head)

# G1 Step 4: Unemployment Trends Visualization
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='unemployment_rate', label='Unemployment Rate', color='b')
plt.axvline(pd.to_datetime("2020-03-01"), color='r', linestyle="--", label="COVID-19 Start")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate in India Over Time")
plt.legend()
plt.show()

# Step 5: Impact of COVID-19
pre_covid = df[df['date'] < '2020-03-01']
post_covid = df[df['date'] >= '2020-03-01']

pre_avg = pre_covid['unemployment_rate'].mean()
post_avg = post_covid['unemployment_rate'].mean()

print(f"\nAverage Unemployment Rate Before COVID-19: {pre_avg:.2f}%")
print(f"Average Unemployment Rate After COVID-19: {post_avg:.2f}%")

#G2  Step 6: State-wise Unemployment Analysis
statewise_avg = df.groupby('region')['unemployment_rate'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=statewise_avg, x='unemployment_rate', y='region', palette='coolwarm')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("State")
plt.title("State-wise Average Unemployment Rate in India")
plt.show()


# G3 Step 8: Labour Participation Rate Analysis
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='labour_participation_rate', label='Labour Participation Rate', color='g')
plt.axvline(pd.to_datetime("2020-03-01"), color='r', linestyle="--", label="COVID-19 Start")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")
plt.title("Labour Participation Rate Over Time")
plt.legend()
plt.show()

# G4 Step 9: Area-wise Unemployment Analysis
areawise_avg = df.groupby('area')['unemployment_rate'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=areawise_avg, x='unemployment_rate', y='area', palette='coolwarm')
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Area (Rural / Urban)")
plt.title("Area-wise Average Unemployment Rate in India")
plt.show()


#G5 step 10: Unemployment Rate by Region Over Time (Grouped Line Plot)
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='date', y='unemployment_rate', hue='region', legend=False, alpha=0.6)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate Across Different Regions Over Time")
plt.show()

#G6 step 11: Boxplot of Unemployment Rate Across Regions (To Compare Distributions)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='unemployment_rate', y='region', palette='coolwarm')
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")
plt.title("Distribution of Unemployment Rate Across Regions")
plt.show()

# G7 step 12: Heatmap of Unemployment Rate Over Time (Monthly Trends)
df['year_month'] = df['date'].dt.to_period("M")
pivot = df.pivot_table(values='unemployment_rate', index='region', columns='year_month', aggfunc='mean')

plt.figure(figsize=(14, 7))
sns.heatmap(pivot, cmap="coolwarm", annot=False)
plt.xlabel("Year-Month")
plt.ylabel("Region")
plt.title("Unemployment Rate Heatmap (Monthly Trends)")
plt.show()

#G8 step 13: Violin Plot of Labour Participation Rate by Region
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='region', y='labour_participation_rate', palette='Set2')
plt.xticks(rotation=90)
plt.xlabel("Region")
plt.ylabel("Labour Participation Rate (%)")
plt.title("Labour Participation Rate Distribution Across Regions")
plt.show()

# G9 step 14: Scatter Plot Between Estimated Employed & Unemployment Rate
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='estimated_employed', y='unemployment_rate', alpha=0.5)
plt.xlabel("Estimated Employed")
plt.ylabel("Unemployment Rate (%)")
plt.title("Relation Between Estimated Employment & Unemployment Rate")
plt.show()

# G10 step15 : KDE Plot for Unemployment Rate by Area
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x='unemployment_rate', hue='area', fill=True, alpha=0.4, palette='coolwarm')
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Density")
plt.title("Distribution of Unemployment Rate (Rural vs. Urban)")
plt.legend(title="Area")
plt.show()

#G11 step 16: Pairplot to visualize relationships between numerical variables
sns.pairplot(df, vars=['unemployment_rate', 'estimated_employed', 'labour_participation_rate'], hue='area', palette='coolwarm')
plt.suptitle("Pairplot of Unemployment Data", y=1.02)
plt.show()

#G12 step 17: Boxplot of Unemployment Rate by Area
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='area', y='unemployment_rate', palette='coolwarm')
plt.xlabel("Area (Rural / Urban)")
plt.ylabel("Unemployment Rate (%)")
plt.title("Boxplot of Unemployment Rate by Area")
plt.show()

#G13 step 18: Extract month for analysis
df['month'] = df['date'].dt.month

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='month', y='unemployment_rate', estimator='mean', ci=None, marker="o", color='b')
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.title("Monthly Trend in Unemployment Rate")
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

#G14 step 19: Violin Plot for Labour Participation Rate by Area
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='area', y='labour_participation_rate', palette='coolwarm')
plt.xlabel("Area (Rural / Urban)")
plt.ylabel("Labour Participation Rate (%)")
plt.title("Distribution of Labour Participation Rate by Area")
plt.show()

#G15 step 20: Scatter Plot of Unemployment Rate vs Labour Participation Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='labour_participation_rate', y='unemployment_rate', hue='area', alpha=0.6, palette='coolwarm')
plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate vs Labour Participation Rate")
plt.legend(title="Area")
plt.show()
