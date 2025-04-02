from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained pipeline model
pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        pclass = int(request.form.get('pclass'))
        sex = request.form.get('sex')
        age = float(request.form.get('age'))
        sibsp = int(request.form.get('sibsp'))
        parch = int(request.form.get('parch'))
        fare = float(request.form.get('fare'))
        embarked = request.form.get('embarked')
        
        # Convert categorical values if necessary (adjust based on model preprocessing)
        sex = 1 if sex == 'male' else 0
        embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}  # Adjust if needed
        embarked = embarked_mapping.get(embarked, 2)
        
        # Prepare input for prediction
        test_input = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]], dtype=object)
        prediction = pipe.predict(test_input)
        
        # Convert prediction result
        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
