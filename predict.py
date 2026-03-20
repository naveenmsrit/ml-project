import joblib
import pandas as pd

# load model
model = joblib.load("house_model.pkl")

# input data
data = pd.DataFrame({
    "area":[2200],
    "bedrooms":[3],
    "age":[10]
})

prediction = model.predict(data)

print("Predicted House Price:", prediction[0])