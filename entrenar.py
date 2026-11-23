import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("📂 1. Cargando conjunto de datos desde 'alumnos.csv'...")

try:
    df = pd.read_csv('alumnos.csv', sep=',')
    # Si toda la fila quedó agrupada en una sola columna, dividirla automáticamente
    if len(df.columns) == 1 and ',' in df.columns[0]:
        col_unica = df.columns[0]
        nuevas_cols = [c.strip().lower() for c in col_unica.split(',')]
        df = df[col_unica].str.split(',', expand=True)
        df.columns = nuevas_cols
        df = df.apply(pd.to_numeric, errors='coerce')
    else:
        df.columns = df.columns.str.strip().str.lower()
    print(f"✅ Archivo cargado correctamente. Registros totales: {len(df)}")
    print(f"📋 Columnas detectadas: {df.columns.tolist()}")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'alumnos.csv'. Revisa la ruta.")
    exit()

# Limpieza: eliminar filas incompletas
df = df.dropna()

# Verificar columnas requeridas
columnas_requeridas = ['asistencia', 'promedio', 'horas_estudio', 'riesgo']
if not all(col in df.columns for col in columnas_requeridas):
    print(f"❌ Error: Tu CSV debe tener las columnas: {columnas_requeridas}")
    exit()

# Entradas (X) y salida (y)
X = df[['asistencia', 'promedio', 'horas_estudio']]
y = df['riesgo']

# 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 2. Entrenando el algoritmo RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

joblib.dump(modelo, 'modelo_riesgo.pkl')
print("✅ 3. ¡Éxito! Modelo exportado como 'modelo_riesgo.pkl'.")