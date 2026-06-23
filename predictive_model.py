import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

print("Generating local dataset internally...")
np.random.seed(42)
num_samples = 5000

data = {
    'Avg. Area Income': np.random.normal(70000, 10000, num_samples),
    'Avg. Area House Age': np.random.normal(6, 1, num_samples),
    'Avg. Area Number of Rooms': np.random.normal(7, 1, num_samples),
    'Avg. Area Number of Bedrooms': np.random.normal(4, 1, num_samples),
    'Area Population': np.random.normal(35000, 10000, num_samples),
}

data['Price'] = (
    data['Avg. Area Income'] * 21 +
    data['Avg. Area House Age'] * 165000 +
    data['Avg. Area Number of Rooms'] * 120000 +
    data['Area Population'] * 15 +
    np.random.normal(0, 50000, num_samples)
)

df = pd.DataFrame(data)

X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

r2_score = model.score(X_test, y_test)
print(f"Model R2 Score: {r2_score:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, model.predict(X_test), alpha=0.3, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted House Prices')
plt.savefig('predictions_plot.png') 
plt.show()