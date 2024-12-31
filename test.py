import pickle
import numpy as np

# Load the trained model and PowerTransformer
with open('D:\\EDA\\CarPrice_Predictor v1.0\\FordCarPrice_Predictor.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['Model']
    transformer = data['Transformer']
    scaler = data['Scaler']

# Test data
test_data = [[6, 1, 3, 2018, 9083, 150, 57.7, 1]]

transformed_test_data = transformer.transform([test_data[0][3:]])
scaled_test_data = scaler.transform(transformed_test_data)

print(transformed_test_data, scaled_test_data)
print(list(transformed_test_data[0]))
print(list(scaled_test_data[0]))
print(np.concatenate([test_data[0][:3], scaled_test_data[0]]))

predicted_car_price = model.predict([np.concatenate([test_data[0][:3], scaled_test_data[0]])])

# Display the result
print(f"Predicted Car Price: {predicted_car_price[0]:.2f}")
