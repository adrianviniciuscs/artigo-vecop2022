import joblib
model = joblib.load(r"../MLWebApp/rfc_model.pkl")


def prediction_model(ejection_fraction, serum_creatinine):
    prediction = model.predict([[ejection_fraction, serum_creatinine]])
    return prediction


def prediction_model_proba(ejection_fraction, serum_creatinine):
    proba = model.predict_proba([[ejection_fraction, serum_creatinine]])
    return proba
