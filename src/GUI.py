from flask import Flask, render_template,request
import training

app = Flask(__name__)

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        text = request.form['inputName']
        x=text
        output=training.main(string=x,option='0')
        return render_template("second.html",result = output)
    
@app.route('/')
def hello_world():
    return render_template('first.html')

if __name__ == '__main__':
   app.run()
