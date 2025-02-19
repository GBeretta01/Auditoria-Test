import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configurar la página
st.set_page_config(page_title="Detección de Fraude", layout="wide")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("datos_fraude.csv")

data = cargar_datos()
variables = ['Total_Inventario', 'Total_CxC', 'Total_CxP', 'Total_Ingresos', 'Total_Gastos']

# Sidebar: Controles
st.sidebar.header("Configuración")
contamination = st.sidebar.slider("Sensibilidad (% Anomalías)", 1, 20, 5) / 100
selected_vars = st.sidebar.multiselect("Variables para Análisis", variables, default=variables[:2])

# Procesamiento
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[variables])
data_scaled = pd.DataFrame(data_scaled, columns=variables)
data = pd.concat([data, data_scaled.add_suffix('_Scaled')], axis=1)

model = IsolationForest(contamination=contamination, random_state=42)
model.fit(data_scaled)
data['Score_Anomalia'] = model.decision_function(data_scaled)
data['Riesgo'] = np.where(data['Score_Anomalia'] < np.percentile(data['Score_Anomalia'], 100*contamination), 
                        'Alto Riesgo', 'Bajo Riesgo')

# Dividir en pestañas
tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Relaciones", "Tendencias", "Análisis Profundo"])

with tab1:
    st.header("🔍 Resumen de Riesgo")
    st.dataframe(data[['Entidad', 'Año', 'Riesgo'] + variables].sort_values('Riesgo'), height=500)

with tab2:
    st.header("📊 Matriz de Relaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlaciones (Pearson)")
        corr_matrix = data[variables].corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(x="Variable", y="Variable", color="Correlación"),
                       x=variables, y=variables,
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("PairPlot Interactivo")
        fig = px.scatter_matrix(data, 
                                dimensions=selected_vars,
                                color="Riesgo",
                                hover_name="Entidad",
                                opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("📈 Tendencias Temporales")
    
    empresa = st.selectbox("Seleccionar Empresa", data['Entidad'].unique())
    df_empresa = data[data['Entidad'] == empresa]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolución Absoluta")
        fig = px.line(df_empresa, x='Año', y=variables, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Evolución Estandarizada")
        fig = px.line(df_empresa, x='Año', y=[f'{v}_Scaled' for v in variables], markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("🔎 Análisis Multivariante")
    
    st.subheader("Distribuciones por Riesgo")
    fig = px.violin(data, x="Riesgo", y=selected_vars, box=True, points="all")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("PCA 3D (Anomalías)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(data_scaled)
    fig = px.scatter_3d(components, x=0, y=1, z=2, 
                        color=data['Riesgo'],
                        labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
                        title="Reducción de Dimensionalidad (PCA)")
    st.plotly_chart(fig, use_container_width=True)

# Mensaje final
st.sidebar.success("Configuración aplicada. Explora las relaciones en las pestañas!")