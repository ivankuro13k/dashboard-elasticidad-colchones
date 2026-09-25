import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
n_semanas = 104

descuento_pct = np.random.uniform(5, 35, n_semanas)
inversion_publicidad = np.random.uniform(2, 10, n_semanas)
precio_base = 1800000

precio_venta = precio_base * (1 - descuento_pct / 100) + np.random.normal(0, 15000, n_semanas)
unidades_vendidas = (
    140 
    - 54.9 * (precio_venta / 1e6) 
    + 8.5 * descuento_pct 
    + 5.0 * np.log(inversion_publicidad) 
    + np.random.normal(0, 3, n_semanas)
)

df = pd.DataFrame({
    'Semana': np.arange(1, n_semanas + 1),
    'Precio_Venta_M': precio_venta / 1e6,
    'Pct_Descuento': descuento_pct,
    'Inversion_Publicidad_M': inversion_publicidad,
    'Unidades_Vendidas': unidades_vendidas,
    'Alta_Rotacion': (unidades_vendidas > np.median(unidades_vendidas)).astype(int)
})

df.to_csv("data/datos_colchones_procesados.csv", index=False)

# Modelo Lineal
X_reg = df[['Precio_Venta_M', 'Pct_Descuento', 'Inversion_Publicidad_M']]
y_reg = df['Unidades_Vendidas']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_reg)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_poly, y_reg, test_size=0.25, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

grid_ridge = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=5, scoring='r2')
grid_ridge.fit(X_train_r_scaled, y_train_r)
best_ridge = grid_ridge.best_estimator_

# Modelo Logístico
X_log = df[['Pct_Descuento', 'Precio_Venta_M', 'Inversion_Publicidad_M']]
y_log = df['Alta_Rotacion']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_log, y_log, test_size=0.25, random_state=42)

scaler_l = StandardScaler()
X_train_l_scaled = scaler_l.fit_transform(X_train_l)
X_test_l_scaled = scaler_l.transform(X_test_l)

grid_log = GridSearchCV(LogisticRegression(random_state=42), {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs']}, cv=5, scoring='accuracy')
grid_log.fit(X_train_l_scaled, y_train_l)
best_log = grid_log.best_estimator_

joblib.dump({'model': best_ridge, 'scaler': scaler_r, 'poly': poly}, "models/modelo_lineal_optimizado.pkl")
joblib.dump({'model': best_log, 'scaler': scaler_l}, "models/modelo_logistico_optimizado.pkl")

print("✅ Modelos optimizados y datos generados con éxito.")