import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

# 1. Cargar datos y modelos
df = pd.read_csv("data/datos_colchones_procesados.csv")
model_lin_data = joblib.load("models/modelo_lineal_optimizado.pkl")
model_log_data = joblib.load("models/modelo_logistico_optimizado.pkl")

# 2. Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Evitar traducción automática del navegador que altera los IDs de los callbacks
app.index_string = '''
<!DOCTYPE html>
<html lang="es" class="notranslate" translate="no">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# 3. Definir el Layout de la aplicación
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Dashboard Estratégico de Elasticidad de Precios", className="text-center text-primary mt-3"),
            html.P("Colchones El Dorado - Canal Retail", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulador Comercial", className="bg-primary text-white font-weight-bold"),
                dbc.CardBody([
                    html.Label("Porcentaje de Descuento (%):", className="fw-bold"),
                    dcc.Slider(
                        id='slider-desc',
                        min=5, max=35, step=1, value=10,
                        marks={i: f"{i}%" for i in range(5, 36, 5)}
                    ),
                    html.Br(),
                    html.Label("Inversión Publicitaria ($ Millones):", className="fw-bold"),
                    dcc.Slider(
                        id='slider-pub',
                        min=2, max=10, step=0.5, value=8,
                        marks={i: f"${i}M" for i in range(2, 11, 2)}
                    ),
                    html.Br(),
                    html.Label("Precio Base de Lista ($ Millones):", className="fw-bold"),
                    dcc.Input(id='input-precio-base', type='text', value='1.8', className="form-control"),
                    html.Hr(),
                    html.Div(id='resultado-simulacion', className="p-3 bg-light rounded border")
                ])
            ], className="shadow-sm mb-4")
        ], md=4),

        dbc.Col([
            dbc.Tabs([
                dbc.Tab(dcc.Graph(id='grafico-elasticidad'), label="Elasticidad Precio vs Ventas"),
                dbc.Tab(dcc.Graph(id='grafico-probabilidad'), label="Probabilidad de Alta Rotación")
            ])
        ], md=8)
    ])
], fluid=True)

# 4. Callback para interactividad en tiempo real
@app.callback(
    [Output('resultado-simulacion', 'children'),
     Output('grafico-elasticidad', 'figure'),
     Output('grafico-probabilidad', 'figure')],
    [Input('slider-desc', 'value'),
     Input('slider-pub', 'value'),
     Input('input-precio-base', 'value')]
)
def simular_y_actualizar(desc, pub, precio_base_input):
    # Validar y convertir precio base (soporta coma y punto decimal)
    try:
        if isinstance(precio_base_input, str):
            precio_base_input = precio_base_input.replace(',', '.')
        precio_base_m = float(precio_base_input)
        if precio_base_m <= 0:
            precio_base_m = 1.8
    except (TypeError, ValueError):
        precio_base_m = 1.8

    if desc is None: desc = 10
    if pub is None: pub = 8

    precio_final_m = precio_base_m * (1 - desc / 100)

    # Predicción Modelo Lineal (Ventas)
    X_input_lin = pd.DataFrame({
        'Precio_Venta_M': [precio_final_m],
        'Pct_Descuento': [desc],
        'Inversion_Publicidad_M': [pub]
    })
    X_poly = model_lin_data['poly'].transform(X_input_lin)
    X_scaled = model_lin_data['scaler'].transform(X_poly)
    pred_ventas = float(model_lin_data['model'].predict(X_scaled)[0])

    # Predicción Modelo Logístico (Probabilidad de Alta Rotación)
    X_input_log = pd.DataFrame({
        'Pct_Descuento': [desc],
        'Precio_Venta_M': [precio_final_m],
        'Inversion_Publicidad_M': [pub]
    })
    X_log_scaled = model_log_data['scaler'].transform(X_input_log)
    prob_rotacion = float(model_log_data['model'].predict_proba(X_log_scaled)[0][1] * 100)

    res_html = [
        html.H5("Resultados Estimados:"),
        html.P([html.Strong("Precio Venta: "), f"${precio_final_m:.2f} M COP"]),
        html.P([html.Strong("Ventas Estimadas: "), f"{pred_ventas:.1f} Unidades"]),
        html.P([html.Strong("Probabilidad Alta Rotación: "), f"{prob_rotacion:.1f}%"])
    ]

    # Gráfico 1: Scatter Plot Elasticidad
    fig_elast = px.scatter(
        df, x='Precio_Venta_M', y='Unidades_Vendidas', color='Pct_Descuento',
        title="Elasticidad Precio vs Ventas", template='plotly_white',
        labels={
            'Precio_Venta_M': 'Precio de Venta ($ Millones COP)',
            'Unidades_Vendidas': 'Unidades Vendidas',
            'Pct_Descuento': 'Descuento (%)'
        }
    )
    fig_elast.add_trace(
        go.Scatter(
            x=[precio_final_m], y=[pred_ventas], mode='markers',
            marker=dict(color='red', size=14, symbol='diamond', line=dict(width=2, color='black')),
            name='Simulación'
        )
    )
    fig_elast.update_layout(margin=dict(r=80, t=80))

    # Gráfico 2: Curva de Probabilidad
    descuentos_range = np.linspace(5, 35, 50)
    precios_range = precio_base_m * (1 - descuentos_range / 100)
    df_curva = pd.DataFrame({
        'Pct_Descuento': descuentos_range,
        'Precio_Venta_M': precios_range,
        'Inversion_Publicidad_M': [pub]*50
    })
    df_curva_scaled = model_log_data['scaler'].transform(df_curva)
    probs_curva = model_log_data['model'].predict_proba(df_curva_scaled)[:, 1] * 100

    fig_prob = px.line(
        x=descuentos_range, y=probs_curva,
        labels={'x': 'Descuento (%)', 'y': 'Probabilidad (%)'},
        title="Curva de Probabilidad de Alta Rotación", template='plotly_white'
    )
    fig_prob.add_vline(x=desc, line_dash="dash", line_color="red")

    return res_html, fig_elast, fig_prob

if __name__ == '__main__':
    app.run(debug=True)