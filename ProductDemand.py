from flask import Flask, jsonify, request, render_template
import pickle


with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        total_price = float(request.form['total_price'])
        base_price = float(request.form['base_price'])

        y_pred = model.predict([[total_price, base_price]])

        return render_template('index.html', prediction='{}'.format(round(y_pred[0])))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)



