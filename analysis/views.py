import matplotlib
matplotlib.use('Agg')

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score


# ---------------- REGISTER ----------------
def register_view(request):
    error = None

    if request.method == "POST":
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            error = "Username already exists"
        else:
            User.objects.create_user(username=username, email=email, password=password)
            return redirect('/')

    return render(request, 'analysis/register.html', {"error": error})


# ---------------- LOGIN ----------------
def login_view(request):
    error = None

    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('/dashboard/')
        else:
            error = "Invalid credentials"

    return render(request, 'analysis/login.html', {"error": error})


# ---------------- LOGOUT ----------------
def logout_view(request):
    logout(request)
    return redirect('/')


# ---------------- DASHBOARD ----------------
def home(request):

    context = {}

    if request.method == "POST" and request.FILES.get('dataset'):

        df = pd.read_csv(request.FILES['dataset'])

        context['data'] = df.head().to_html(classes='table table-bordered', index=False)
        context['rows'] = df.shape[0]
        context['cols'] = df.shape[1]
        context['missing'] = int(df.isnull().sum().sum())

        summary = df.describe().T
        context['summary'] = summary.to_html(classes='table table-bordered')

        os.makedirs("analysis/static", exist_ok=True)

        numeric_df = df.select_dtypes(include=['number'])

        # ---------------- MODEL TRAINING ----------------
        if numeric_df.shape[1] > 1:

            X = numeric_df.iloc[:, :-1]
            y = numeric_df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            if y.nunique() <= 10:
                model = RandomForestClassifier()
                score_func = accuracy_score
                model_type = "Classification"
            else:
                model = RandomForestRegressor()
                score_func = r2_score
                model_type = "Regression"

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            context['accuracy'] = round(score_func(y_test, preds) * 100, 2)
            context['model_type'] = model_type

            joblib.dump(model, "analysis/model.pkl")

            # ---------------- FEATURE IMPORTANCE ----------------
            importance = model.feature_importances_

            imp_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values(by='Importance')

            plt.figure()
            sns.barplot(x='Importance', y='Feature', data=imp_df)
            plt.title("Feature Importance")
            plt.savefig("analysis/static/feature.png")
            plt.close()

            context['feature'] = "feature.png"

        # ---------------- HISTOGRAM ----------------
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0:
            plt.figure()
            df[numeric_cols[0]].hist()
            plt.title("Histogram")
            plt.savefig("analysis/static/hist.png")
            plt.close()

            context['hist'] = "hist.png"

        # ---------------- CORRELATION HEATMAP ----------------
        if len(numeric_cols) >= 2:

            plt.figure(figsize=(6,4))

            sns.heatmap(
                df[numeric_cols].corr(),
                cmap="coolwarm",
                annot=True
            )

            plt.title("Correlation Heatmap")

            plt.savefig("analysis/static/heatmap.png")
            plt.close()

            context['heatmap'] = "heatmap.png"

    return render(request, 'analysis/home.html', context)