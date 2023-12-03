from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

# Cargar el modelo entrenado al iniciar la aplicación
modelo_entrenado = joblib.load('modelo_entrenado.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener datos del formulario
        edad = request.form['edad']
        sexo = request.form['sexo']
        fuma = request.form['fuma']
        diabetes = request.form['diabetes']
        presion = request.form['presion']
        anemia = request.form['anemia']
        creatina = request.form['creatina']
        fraccion = request.form['fraccion']
        plaqueta = request.form['plaqueta']
        creatinina = request.form['creatinina']
        sodio = request.form['sodio']
        tiempo = request.form['tiempo']

        # Realizar predicción con el modelo
        # (Asegúrate de preprocesar los datos de entrada de la misma manera que durante el entrenamiento)
        datos_entrada = np.array([[
            edad,
            anemia,
            creatina,
            diabetes,
            fraccion,
            presion,
            plaqueta,
            creatinina,
            sodio,
            sexo,
            fuma,
            tiempo, 
        ]])

        # Ajusta esta línea según tu necesidad
        resultado_prediccion = modelo_entrenado.predict(datos_entrada)
        resultado_prediccion = np.argmax(resultado_prediccion, axis=1)

        # Puedes devolver el resultado como JSON
        return jsonify({'resultado_prediccion': resultado_prediccion.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
