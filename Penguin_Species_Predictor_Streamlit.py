import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title(":blue[Cracking the Code of Penguins: Your Species Predictor 🐧]")
st.image('Penguin.jpeg')
# Load dataset
df = pd.read_csv("penguins_size.csv")

# Drop nulls
df.dropna(inplace=True)

# Encode target
y = df.pop("species")
X = df.copy()

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate categorical and numerical data
num_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_cols = ['island', 'sex']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = StandardScaler()

# Encode training data
X_train_cat = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)
Final_X_train = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)

# Encode testing data
X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)
Final_X_test = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

# KNN Classifier
knn = KNeighborsClassifier()
knn.fit(Final_X_train, y_train)
y_pred = knn.predict(Final_X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc:.2f}")

# UI input fields
culmen_length = st.number_input("Enter Culmen Length (mm)")
culmen_depth = st.number_input("Enter Culmen Depth (mm)")
flipper_length = st.number_input("Enter Flipper Length (mm)")
body_mass = st.number_input("Enter Body Mass (g)")

islands = sorted(df['island'].unique())
selected_island = st.selectbox("Select Island", islands)

sexes = sorted(df['sex'].dropna().unique())
selected_sex = st.selectbox("Select Sex", sexes)

if st.button("Predict Penguin Species"):
    query_df = pd.DataFrame([{
        'culmen_length_mm': culmen_length,
        'culmen_depth_mm': culmen_depth,
        'flipper_length_mm': flipper_length,
        'body_mass_g': body_mass,
        'island': selected_island,
        'sex': selected_sex
    }])

    query_cat = encoder.transform(query_df[cat_cols])
    query_num = scaler.transform(query_df[num_cols])

    query_final = pd.concat([
        pd.DataFrame(query_num, columns=num_cols),
        pd.DataFrame(query_cat, columns=encoder.get_feature_names_out(cat_cols))
    ], axis=1)

    prediction = knn.predict(query_final)[0]
    st.success(f"Predicted Penguin Species: {prediction}")