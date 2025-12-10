# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# ----------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------
st.set_page_config(page_title="MAESTRO DE NOTAS", page_icon="🎓", layout="wide")

# Temas visuales
theme = st.sidebar.radio("Tema visual", ["Claro", "Oscuro"], index=1)

if theme == "Oscuro":
    st.markdown("""
        <style>
            body, .stApp { background-color: #0f172a; color: #e2e8f0; }
            .metric-card {background:#1e293b !important; color:#e2e8f0 !important;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body, .stApp { background-color: #f8fafc; color: #0f172a; }
            .metric-card {background:#e2e8f0 !important; color:#0f172a !important;}
        </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# CARGA DE DATOS
# ----------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

st.sidebar.header("Parámetros de Datos")
data_path = st.sidebar.text_input("Ruta del dataset", "data/academic_performance_master.csv")

df = load_data(data_path)

# FIX total para Arrow
obj_cols = df.select_dtypes(include=["object"]).columns
df[obj_cols] = df[obj_cols].astype(str)

st.title("🎓 Maestro de Notas — Exploración, Clasificación y Clustering")

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Exploración",
    "Clasificación",
    "Clustering",
    "Outliers & Reportes"
])

# ----------------------------------------------------
# TAB 1 – EXPLORACIÓN
# ----------------------------------------------------
with tab1:
    st.header("📊 Exploración Inicial")
    st.write(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", df.shape[0])
    col2.metric("Columnas", df.shape[1])
    col3.metric("Nulos", int(df.isnull().sum().sum()))

    st.subheader("Distribuciones clave")
    for v in ["Nota_final", "Asistencia"]:
        if v in df:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[v].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

# ----------------------------------------------------
# TAB 2 – CLASIFICACIÓN
# ----------------------------------------------------
with tab2:
    st.header("🤖 Modelo Supervisado")

    umbral = st.slider("Umbral de aprobado", 0, 100, 70)
    df["aprobado"] = (df["Nota_final"] >= umbral).astype(int)

    numericas = ["Nota_final", "Asistencia", "Parciales", "Tareas"]
    numericas = [c for c in numericas if c in df.columns]

    categ = ["Nivel", "Carrera"]
    categ = [c for c in categ if c in df.columns]

    X = df[numericas + categ]
    y = df["aprobado"]

    if y.nunique() < 2:
        st.error("Solo existe una clase. Ajusta el umbral.")
    else:

        preprocess = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numericas),

            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categ)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        C = st.sidebar.slider("C Regularización", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])

        clf = Pipeline([
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=1000, C=C, solver=solver))
        ])

        if st.button("Entrenar Modelo"):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            st.metric("Accuracy", f"{acc:.3f}")

            st.subheader("Matriz de confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
            st.pyplot(fig)

            # Importancia de variables
            st.subheader("📌 Interpretación Automática del Modelo")

            ohe = clf.named_steps["prep"].named_transformers_["cat"]
            cat_names = list(ohe.get_feature_names_out(categ)) if categ else []

            feature_names = numericas + cat_names
            coef = clf.named_steps["model"].coef_[0]

            imp = pd.Series(coef, index=feature_names).sort_values(key=lambda x: abs(x), ascending=False)

            st.write("### Variables más influyentes")
            st.write(imp.head(10))

            # Interpretación textual automática
            st.write("### 🔍 Interpretación del modelo (explicada)")
            for var, val in imp.head(5).items():
                if val > 0:
                    st.write(f"✔️ **{var}** aumenta la probabilidad de aprobar.")
                else:
                    st.write(f"⚠️ **{var}** disminuye la probabilidad de aprobar.")

# ----------------------------------------------------
# TAB 3 – CLUSTERING
# ----------------------------------------------------
with tab3:
    st.header("🌀 Clustering (K-Means)")
    cluster_features = ["Asistencia", "Nota_final"]

    if all(c in df for c in cluster_features):
        k = st.slider("Número de clusters", 2, 5, 3)
        scale = st.checkbox("Escalar variables", True)

        Xc = df[cluster_features].dropna()
        scaler = StandardScaler()
        Z = scaler.fit_transform(Xc) if scale else Xc.values

        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(Z)

        df_cluster = Xc.copy()
        df_cluster["cluster"] = labels

        st.write(df_cluster.head())

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df_cluster, x="Asistencia", y="Nota_final",
                        hue="cluster", palette="Set2", s=60, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("El dataset debe tener Asistencia y Nota_final.")

# ----------------------------------------------------
# TAB 4 – OUTLIERS + REPORTES
# ----------------------------------------------------
with tab4:
    st.header("📌 Detección Automática de Outliers + Descargas")

    if "Nota_final" in df and "Asistencia" in df:

        st.subheader("Outliers en Nota_final y Asistencia")

        def detectar_outliers(col):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR
            mascara = (df[col] < limite_inf) | (df[col] > limite_sup)
            return df[mascara]

        out_notas = detectar_outliers("Nota_final")
        out_asist = detectar_outliers("Asistencia")

        st.write("### Outliers en Notas")
        st.write(out_notas)

        st.write("### Outliers en Asistencia")
        st.write(out_asist)

        # Botón descargar CSV
        st.subheader("📥 Descargar reporte CSV")

        buffer_csv = io.StringIO()
        df.to_csv(buffer_csv, index=False)
        st.download_button("Descargar dataset completo (.csv)", buffer_csv.getvalue(), "reporte.csv", "text/csv")

        # Botón descargar PDF
        st.subheader("📥 Descargar reporte PDF")

        import matplotlib.backends.backend_pdf as pdf_backend

        pdf_buffer = io.BytesIO()
        pdf = pdf_backend.PdfPages(pdf_buffer)

        fig, ax = plt.subplots()
        sns.boxplot(data=df[["Nota_final", "Asistencia"]], ax=ax)
        ax.set_title("Distribución de Nota_final y Asistencia")
        pdf.savefig(fig)

        pdf.close()
        st.download_button("Descargar reporte PDF", pdf_buffer.getvalue(), file_name="reporte.pdf")
