from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np


with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('lasso_regression_model.pkl', 'rb') as f:
    mod = pickle.load(f)
with open('ridge_regression_model.pkl', 'rb') as f:
    mod2 = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    base_price = 0
    q1 = base_price * 0.25
    q2 = base_price * 0.5
    q3 = base_price * 0.75
    q4 = base_price
    q5 = base_price * 1.25
    q6 = base_price * 1.5
    q7 = base_price * 1.75
    y_pred_q1 = model.predict([[q1, base_price]])
    y_pred_q2 = model.predict([[q2, base_price]])
    y_pred_q3 = model.predict([[q3, base_price]])
    y_pred_q4 = model.predict([[q4, base_price]])
    y_pred_q5 = model.predict([[q5, base_price]])
    y_pred_q6 = model.predict([[q6, base_price]])
    y_pred_q7 = model.predict([[q7, base_price]])
    if request.method == 'POST':
        base_price = float(request.form['base_price'])

        # Calculate quartiles
        q1 = base_price * 0.25
        q2 = base_price * 0.5
        q3 = base_price * 0.75
        q4 = base_price
        q5 = base_price * 1.25
        q6 = base_price * 1.5
        q7 = base_price * 1.75

        y_pred_q1 = model.predict([[q1, base_price]])
        y_pred_q2 = model.predict([[q2, base_price]])
        y_pred_q3 = model.predict([[q3, base_price]])
        y_pred_q4 = model.predict([[q4, base_price]])
        y_pred_q5 = model.predict([[q5, base_price]])
        y_pred_q6 = model.predict([[q6, base_price]])
        y_pred_q7 = model.predict([[q7, base_price]])

        return render_template('index.html', rate1=q1, rate2=q2, rate3=q3, rate4=q4, rate5=q5, rate6=q6, rate7=q7, q1=round(y_pred_q1[0]), median=round(y_pred_q2[0]), q3=round(y_pred_q3[0]), q4=round(y_pred_q4[0]), q5=round(y_pred_q5[0]), q6=round(y_pred_q6[0]), q7=round(y_pred_q7[0]))

    return render_template('index.html', rate1=q1, rate2=q2, rate3=q3, rate4=q4, rate5=q5, rate6=q6, rate7=q7, q1=round(y_pred_q1[0]), median=round(y_pred_q2[0]), q3=round(y_pred_q3[0]), q4=round(y_pred_q4[0]), q5=round(y_pred_q5[0]), q6=round(y_pred_q6[0]), q7=round(y_pred_q7[0]))


if __name__ == '__main__':
    app.run(debug=True)
