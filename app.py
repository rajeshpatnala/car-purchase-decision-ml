from flask import Flask, request, render_template
from flask_cors import cross_origin
from modules.load_clusters import cls_knn, cls_iter
from modules.load_models import mdl_knn, mdl_iter
from modules.transformation import ip_transform, op_transform
import pandas as pd

app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_method():
         
    if request.method == "POST":

        buying_price = request.form['buying_price']
        maintenance = request.form['maintenance']
        doors = request.form['doors']
        no_of_seaters = request.form['no_of_seaters']
        lug_boot = request.form['lug_boot']
        safety = request.form['safety']

        inputs_str = f"""
                        Buying Price : {buying_price}\n
                        Maintenance : {maintenance}\n
                        Doors : {doors}\n
                        Seater : {no_of_seaters}\n
                        Boot Size : {lug_boot}\n
                        Safety : {safety}\n
                      """

        inputs = [buying_price, maintenance, doors,
                  no_of_seaters, lug_boot, safety]

        tf_inputs = ip_transform(inputs)

        cluster = cls_knn.predict(tf_inputs)
        model = mdl_knn[cluster[0]]
        output = model.predict(tf_inputs)
        
        output = op_transform(output[0])
        return render_template('home.html',inputs = inputs_str, decision=f"Can I Buy Car?\nDecision : {output}") 

    return render_template("home.html")

if __name__=="__main__":
    app.run(debug=True)