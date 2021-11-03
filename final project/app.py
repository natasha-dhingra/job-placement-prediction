import pickle
from flask import Flask,render_template,request

app = Flask(__name__, static_folder="static")
loadedmodel=pickle.load(open('KNNmodel.pkl','rb'))
loadedEncoder = pickle.load(open('Stream Encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    age=request.form['age']
    gender=request.form['gender'] 
    stream=request.form['stream']
    hostel=0
    cgpa=request.form['cgpa']
    backlogs=request.form['backlogs']
    internships=request.form['internships']

    stream = loadedEncoder.transform([stream])[0]

    # prediction=loadedmodel.predict([[age,gender,stream,hostel,cgpa,backlogs,internships]])[0]
    prediction = loadedmodel.predict([[age, internships, cgpa, hostel, backlogs, gender, stream]])[0]

    if prediction==0:
        prediction= "Will Get Placed"
    else:
        prediction= "Won't Get Placed"

    return render_template('index.html',output=prediction)                

if __name__=='__main__':
    app.run(debug=True)
