# Importa las bibliotecas necesarias
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Crea una instancia de la aplicación Flask
app = Flask(__name__)

# Carga el modelo de machine learning
modelo = joblib.load("modelo.pkl")

# Define una ruta para la página principal
@app.route('/')
def home():
    return render_template('index.html')

# Define una ruta para la predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtiene los datos del formulario
    datos = [int(request.form['edad']),
             bool(request.form['anemia']),
             int(request.form['creatina']),
             bool(request.form['diabetes']),
             int(request.form['fraccion']),
             bool(request.form['presion']),
             int(request.form['plaqueta']),
             float(request.form['creatinina']),
             int(request.form['sodio']),
             bool(request.form['sexo']),
             bool(request.form['fuma']),
             int(request.form['tiempo']),
             ]

    # Convierte los datos a un arreglo de NumPy y realiza la predicción
    entrada = np.array(datos).reshape(1, -1)
    prediccion = modelo.predict(entrada)

    # Devuelve la predicción como JSON
    return jsonify({'clase_predicha': str(prediccion[0])})

# Ejecuta la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True)
