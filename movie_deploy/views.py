from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
import pickle
import os
from Model.movieclassifier.vectorizer import vect
import numpy as np

# Create your views here.

def webpage(request):
    return render(request,'deploy.html')

def values(request):
    
    if request.method == 'POST':
        review = [str(request.POST.get('content1'))]
        review1 = str(request.POST.get('content1'))
    
    clf = pickle.load(open(os.path.join('Model/movieclassifier/pkl_objects','classifier.pkl'), 'rb'))
    label = {0:'negative', 1:'positive'}
    #example = ['the movie is fair']
    X = vect.transform(review)
    
    model_pred = label[clf.predict(X)[0]]
    prob = str(round(np.max(clf.predict_proba(X))*100, 2))+"%" 
    
    
    return render(request,'deploy.html', {"get_prediction":model_pred,"get_prob":prob,"review":review1})