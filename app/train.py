from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

X, y = load_diabetes(return_X_y=True, as_frame=True)
feature_names = list(X.columns)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)

joblib.dump({"model": model, "feature_names": feature_names}, "model.joblib")
print("Saved model.joblib with features:", feature_names)
