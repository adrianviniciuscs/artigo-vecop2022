from django.shortcuts import render
import numpy as np
from MLWebApp.ml_model import prediction_model, prediction_model_proba


def home(request):
    return render(request, 'index.html')


def result(request):
    serum_creatinine = float(request.GET["serum_creatinine"])
    ejection_fraction = float(request.GET["ejection_fraction"])
    result = prediction_model(ejection_fraction, serum_creatinine)
    proba = np.max(prediction_model_proba(ejection_fraction, serum_creatinine))
    print(result)
    print(proba)
    return render(request, 'result.html', {'result': result, 'proba': proba})
