from flask import Flask, request, render_template
from model.model_pred import predict_pipeline
from model.model_pred import __version__ as model_version
import os


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def predict():
    # dummy prediction for html template
    prediction = ''
    if request.method == "POST":
        tweet = request.form['tweet']
        prediction = predict_pipeline(tweet)
        # get prediction to display in html page
    return render_template('predict.html', model_version=model_version, prediction=prediction)


# to run the app in a docker and access to it
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)