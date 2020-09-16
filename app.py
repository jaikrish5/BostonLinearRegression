from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))




@app.route('/', methods=['GET', 'POST']) # To render Homepage
def home_page():
    return render_template('index.html')

@app.route('/price', methods=['POST'])  # This will be called from UI
def math_operation():
    if (request.method=='POST'):
        #operation=request.form['operation']
        num1=float(request.form['RM'])
        num2 =float(request.form['Ptr'])
        num3 =float(request.form['status'])
        int_features = []
        int_features.append(num1)
        int_features.append(num2)
        int_features.append(num3)

        print(int_features)

        final_features = [np.array(int_features)]

        print(final_features)



        prediction = model.predict(final_features)

        print(prediction)

        output = round(prediction[0],2)



        # if(operation=='add'):
        #     r=num1+num2
        #     result= 'the sum of '+str(num1)+' and '+str(num2) +' is '+str(r)
        # if (operation == 'subtract'):
        #     r = num1 - num2
        #     result = 'the difference of ' + str(num1) + ' and ' + str(num2) + ' is ' + str(r)
        # if (operation == 'multiply'):
        #     r = num1 * num2
        #     result = 'the product of ' + str(num1) + ' and ' + str(num2) + ' is ' + str(r)
        # if (operation == 'divide'):
        #     r = num1 / num2
        #     result = 'the quotient when ' + str(num1) + ' is divided by ' + str(num2) + ' is ' + str(r)
        return render_template('results.html',result=output)

# @app.route('/via_postman', methods=['POST']) # for calling the API from Postman/SOAPUI
# def math_operation_via_postman():
#     if (request.method=='POST'):
#         operation=request.json['operation']
#         num1=int(request.json['num1'])
#         num2 = int(request.json['num2'])
#         if(operation=='add'):
#             r=num1+num2
#             result= 'the sum of '+str(num1)+' and '+str(num2) +' is '+str(r)
#         if (operation == 'subtract'):
#             r = num1 - num2
#             result = 'the difference of ' + str(num1) + ' and ' + str(num2) + ' is ' + str(r)
#         if (operation == 'multiply'):
#             r = num1 * num2
#             result = 'the product of ' + str(num1) + ' and ' + str(num2) + ' is ' + str(r)
#         if (operation == 'divide'):
#             r = num1 / num2
#             result = 'the quotient when ' + str(num1) + ' is divided by ' + str(num2) + ' is ' + str(r)
#         return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
