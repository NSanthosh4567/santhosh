import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
from pathlib import Path

# Load the CSV relative to this script's directory so the script works
# no matter what the current working directory is when it's executed.
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "shopping.csv"
if not csv_path.exists():
	raise FileNotFoundError(f"Could not find '{csv_path.name}' next to {base_dir!s}. Place the file there or adjust the path.")
shopping = pd.read_csv(csv_path)
print(shopping)
print("Min Max Scaler")
numeric_col=shopping.select_dtypes(include='number').columns
scaler=MinMaxScaler()
shopping_normalized=pd.DataFrame(scaler.fit_transform(shopping[numeric_col]),columns=numeric_col)
print(shopping_normalized.head())
print("Standard Scaler")

numeric_col1=shopping.select_dtypes(include="number").columns
scaler=StandardScaler()
shopping_standardized=pd.DataFrame(scaler.fit_transform(shopping[numeric_col1]),columns=numeric_col1)
print(shopping_standardized.head())

plt.figure(figsize=(8,6))
plt.hist(shopping['Avg_Price'],bins=10)
plt.title("Histogram")
plt.xlabel("Avg_Price")
plt.ylabel("Frequency")
plt.show()