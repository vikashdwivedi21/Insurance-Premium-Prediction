from flask import Flask, request,jsonify, render_template
from src.pipeline.prediction_pieline import CustomData, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods= ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    elif request.method == 'POST':
        data = CustomData(
            age = float(request.form.get('age')),
            sex = str(request.form.get('sex')),
            bmi = float(request.form.get('bmi')),
            children = float(request.form.get('children')),
            smoker = str(request.form.get('smoker')),
            region = str(request.form.get('region')),

        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)
        return render_template('prediction.html',final_result = results)
    
    else:
        return "Invalid Request Method"


if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=True)