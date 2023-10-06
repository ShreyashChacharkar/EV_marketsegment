from flask import Flask, render_template, redirect, url_for, request
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open("model/linear_model.pkl", 'rb'))
with open("model/ev_cars_clean.pkl", 'rb') as file:
     loaded_df = pickle.load(file)
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/submit", methods=["POST", "GET"])
def make_prediction(loaded_model= model, loaded_df =loaded_df):
    Age = float(request.form["Age"])
    profession= str(request.form["Profession"])
    Marrital_Status= str(request.form["Marrital Status"])
    Education= str(request.form["Education"])
    No_of_Dependents= float(request.form["No of Dependents"]) 
    Personal_loan= str(request.form["Personal loan"])
    House_Loan= str(request.form["House loan"])
    wife_working= str(request.form["Wife Working"])
    Total_Salary= float(request.form["Total Salary"])

    predicted = {"Age":Age, 
    "Profession":profession,
    "Marrital Status":Marrital_Status,
    "Education":Education,
    "No of Dependents":No_of_Dependents,
    "Personal loan":Personal_loan,
    "House Loan":House_Loan,
    "Wife Working": wife_working,
    "Total Salary":Total_Salary}

    mapped_dict = {"Yes":1,"No":0,"Married":1,"Single":0,"Post-Graduate":1,"Graduate":0,"Salaried":1, "Business":0}

    for key, value in predicted.items():
        if value in mapped_dict:
            predicted[key] = mapped_dict[value]

    df= pd.DataFrame(predicted, index=[0])

    

    predicted_value = loaded_model.predict(df)[0]
    
    
    loaded_df["predicted"] = ([predicted_value]*len(loaded_df))
    loaded_df["error"] = (loaded_df["predicted"] - loaded_df["Car_price"]).abs()

    df = loaded_df.sort_values("error",ascending=True)

    top_result = df.iloc[0].to_dict()
    name = top_result['Car_name']
    name = name.replace(" ","_")
    predicted_image = f"static/{name}.jpg"

    return render_template("index.html",result = top_result, img=predicted_image)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=8000)