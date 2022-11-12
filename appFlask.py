import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask,render_template,request
app=Flask(__name__)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/result',methods=['GET','POST'])
def Enviar():
    if request.method == "POST":
        try:
            var_1 = float(request.form["P01"])
            var_2 = float(request.form["P02"])
            var_3 = float(request.form["P03"])
            var_4 = float(request.form["P04"])
            var_5 = float(request.form["P05"])
            var_6 = float(request.form["P06"])
            var_7 = float(request.form["P07"])
            var_8 = float(request.form["P08"])
            var_9 = float(request.form["P09"])
            var_10 = float(request.form["P10"])
           
            
            pred_args=[var_1,var_2,var_3,var_4,var_5,var_6,var_7,var_8,var_9,var_10]
            pred_arr=np.array(pred_args)
            preds=pred_arr.reshape(1, -1)
            modelo=open("./modeloNaiveBayes.pkl","rb")
            modelo_class=joblib.load(modelo)
            prediccion_modelo=modelo_class.predict(preds)
            prediccion_modelo=round(float(prediccion_modelo),2)
            if(prediccion_modelo == 1.0):
                prediccion_modelo="Usted posiblemente es alguien que tiene un cuadro de depresión"
                prediccion_m="Si considera que es cierto, acuda a un profesional"
            else:
                prediccion_modelo="No tiene"
                prediccion_m="Si considera que es cierto, este feliz"
        except ValueError:
                return "Por Favor Ingrese Datos Validos"
        return render_template("result.html", prediccion = prediccion_modelo, prediccion2=prediccion_m)


if __name__ == "__main__":
    app.run(debug=True)