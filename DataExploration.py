import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
FILE_PATH = 'cc_institution_details.csv'
data = pd.read_csv(FILE_PATH)

print(data.shape)
print(data.columns)

# view data information
print(
    data.describe(include='all'),
    data.info(),
    data.isnull().sum()
)

corr = data.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()