import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

# Load the dataset
df = pd.read_csv('crime_data.csv')

# Green forest-style color palette
green_palette = sns.color_palette("Greens")

# Step 1: Plot Crime Categories
plt.figure(figsize=(10, 5))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index, palette=green_palette)
plt.title('Crime Category Distribution')
plt.xlabel('Number of Reports')
plt.ylabel('Crime Category')
plt.tight_layout()
plt.show()

# Step 2: Victim Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Victim_Gender', data=df, palette=green_palette)
plt.title('Victim Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Step 3: Offender Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Offender_Gender', data=df, palette=green_palette)
plt.title('Offender Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Step 4: Victim Race Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Victim_Race', data=df, palette=green_palette)
plt.title('Victim Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Step 5: Offender Race Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Offender_Race', data=df, palette=green_palette)
plt.title('Offender Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Machine Learning Section ---

# Clean and prepare data
df_ml = df[['Victim_Age', 'Offender_Age', 'Category']].dropna()

# Linear Regression: Predict Victim Age from Offender Age
X = df_ml[['Offender_Age']]
y = df_ml['Victim_Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nLinear Regression Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot regression line
plt.figure(figsize=(6, 4))
sns.regplot(x='Offender_Age', y='Victim_Age', data=df_ml, scatter_kws={'alpha':0.3}, line_kws={"color": "green"})
plt.title('Linear Regression: Offender Age vs Victim Age')
plt.tight_layout()
plt.show()

# Decision Tree Classifier: Predict Crime Category
# Encode Category as numeric
df_ml['Category_Code'] = df_ml['Category'].astype('category').cat.codes
X = df_ml[['Victim_Age', 'Offender_Age']]
y = df_ml['Category_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

print("\nDecision Tree Classification Report:")
y_pred_tree = tree.predict(X_test)
print(classification_report(y_test, y_pred_tree))

# Plot decision tree
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=['Victim_Age', 'Offender_Age'], class_names=df_ml['Category'].astype('category').cat.categories, filled=True)
plt.title('Decision Tree to Predict Crime Category')
plt.tight_layout()
plt.show()