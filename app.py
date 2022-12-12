from flask import Flask, render_template, url_for, request, redirect

#import the package/Library
import numpy as np
import pandas as pd
import pickle
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/more")
def more():
    return render_template("more.html")

@app.route("/preview")
def preview():
    df = pd.read_csv("milknew.csv")
    return render_template("preview.html", df_view = df)

#creating a function of accuration model
def naive_bayes():
    # membaca data
    df = pd.read_csv("milknew.csv")
    #membuat DataFrame kecuali kolom yang berisi target
    X = df.drop(columns=['Grade'])
    #memisahkan kolom target dan dimasukkan ke dalam variable y
    y = df['Grade'].values
    #melakukan split dataset ke dalam bentuk data train dan data test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    #menggunakan model
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred = gaussian.predict(X_test)
    accuracy=round(accuracy_score(y_test,y_pred)* 100, 2)
    return accuracy

def decision_tree():
    # membaca data
    df = pd.read_csv("milknew.csv")
    #membuat DataFrame kecuali kolom yang berisi target
    X = df.drop(columns=['Grade'])
    #memisahkan kolom target dan dimasukkan ke dalam variable y
    y = df['Grade'].values
    #melakukan split dataset ke dalam bentuk data train dan data test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    #menggunakan model
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    accuracy=round(accuracy_score(y_test,y_pred)* 100, 2)
    return accuracy

def random_forest():
    # membaca data
    df = pd.read_csv("milknew.csv")
    #membuat DataFrame kecuali kolom yang berisi target
    X = df.drop(columns=['Grade'])
    #memisahkan kolom target dan dimasukkan ke dalam variable y
    y = df['Grade'].values
    #melakukan split dataset ke dalam bentuk data train dan data test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    #menggunakan model
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=round(accuracy_score(y_test,y_pred)* 100, 2)
    return accuracy
    
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        pH = request.form['pH']
        Temperature = request.form['Temperature']
        Taste = request.form['Taste']
        Odor = request.form['Odor']
        Fat = request.form['Fat']
        Turbidity = request.form['Turbidity']
        Colour = request.form['Colour']
        model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
        sample_data = [pH,Temperature, Taste, Odor, Fat, Turbidity, Colour]
        clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
        

		# Reloading the Model
        if model_choice == 'naive_bayes':
            with open('nb_pkl', 'rb') as r:
                nb_model = pickle.load(r)
            model = 'Naive Bayes'
            acc = naive_bayes()
            result_prediction = nb_model.predict(ex1)
        elif model_choice == 'decision_tree':
            with open('dt_pkl', 'rb') as r:
                dt_model = pickle.load(r)
            model = 'Decision Tree'
            acc = decision_tree()
            result_prediction = dt_model.predict(ex1)
        elif model_choice == 'random_forest':
            with open('rf_pkl', 'rb') as r:
                rf_model = pickle.load(r)
            model = 'Random Forest'
            acc = random_forest()
            result_prediction = rf_model.predict(ex1)
        else :
            with open('nb_pkl', 'rb') as r:
                nb_model = pickle.load(r)
            model = 'Naive Bayes'
            acc = naive_bayes()
            result_prediction = nb_model.predict(ex1)

    return render_template("index.html",
        accuration = acc,
		result_prediction=result_prediction,
		model_selected=model)

    # return render_template("new.html", pH=pH,Taste=Taste, Fat=Fat, model_selected=model_choice )

@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    Nama_Model= ['Naive Bayes','Decision Tree','Random Forest']
    fig= plt.figure(figsize=(10,7))
    akurasi = [naive_bayes(), decision_tree(), random_forest()]
    plt.bar(Nama_Model, akurasi, color='#0062cc')
    plt.title('Accuracy Result of 3 Methods', size=16)
    plt.ylabel('Accuracy', size=14)
    plt.xlabel('Methods', size=14)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fig

if __name__ == "__main__":
    app.run(debug=True)