import os
import pandas as pd
from joblib import dump
from sklearn.naive_bayes import GaussianNB

DATA_PATH = "data/loan.csv"

def main():
    
    df = pd.read_csv(DATA_PATH)

    # keep only required columns
    df = df[[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Loan_Status"
    ]]

    # remove missing values
    df = df.dropna()

    # features
    X = df[[
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]]

    # target
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    # create model
    model = GaussianNB()

    # train model
    model.fit(X, y)

    # save model
    dump(model, "models/model.joblib")

    print(" Loan model trained and saved!")

if __name__ == "__main__":
    main()