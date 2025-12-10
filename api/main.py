# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

app = FastAPI(title="https://rendimientoacademicotecazuay.streamlit.app")

# Entrenamiento rápido en memoria (en práctica real, carga modelo persistido)
df = pd.read_csv("data/academic_performance_master.csv")
df["aprobado"] = (df["Nota_final"] >= 70).astype(int)
numericas = [c for c in ["Nota_final", "Asistencia", "Parciales", "Tareas"] if c in df.columns]
categoricas = [c for c in ["Nivel", "Carrera"] if c in df.columns]

X = df[numericas + categoricas]
y = df["aprobado"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numericas),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
])
clf = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=1000))])
clf.fit(X, y)

class PredictItem(BaseModel):
    Nota_final: float
    Asistencia: float
    Parciales: float | None = None
    Tareas: float | None = None
    Nivel: str | None = None
    Carrera: str | None = None

class ClusterItem(BaseModel):
    Asistencia: float
    Nota_final: float

@app.post("/predict")
def predict(item: PredictItem):
    row = pd.DataFrame([item.dict()])
    pred = int(clf.predict(row)[0])
    prob = float(clf.predict_proba(row)[0][1])
    return {"aprobado_pred": pred, "prob_aprobado": round(prob, 4)}

@app.post("/cluster")
def cluster(items: list[ClusterItem], k: int = 3):
    dfc = pd.DataFrame([i.dict() for i in items])
    scaler = StandardScaler()
    Z = scaler.fit_transform(dfc[["Asistencia", "Nota_final"]])
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(Z)
    centroids = scaler.inverse_transform(km.cluster_centers_)
    return {
        "labels": labels.tolist(),
        "centroids": centroids.tolist()
    }