#Line Regression Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Mean Squared Error Cost
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = ((1/ (2*m)) * np.sum(np.square(predictions-y)))
    return cost

#Gradient Descent 
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X.dot(theta)
        theta -= (alpha/m) * X.T.dot(predictions - y)
        cost_history.append(compute_cost(X, y, theta))
    
    #print(f"Theta Before Update: {theta}")

    return theta, cost_history

def prep_data(data):
    X = np.array(data[['Month']])
    y = np.array(data['Expense'])
    m = len(y)

    X = np.c_[np.ones(m), X]
    return X, y

def encode_data(df):
    #print("Unique Months in Data:", df['Month'].unique())
    #print("Unique Categories in Data:", df['Category'].unique())
    month_dic = {month: idx + 1 for idx, month in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}

    df['Month'] = df['Month'].map(month_dic)

    category_dic = {"Income": 1, "School": 2, "Retirement": 3, "Utilities": 4, "Entertainment": 5, "Phone payment": 6, "Groceries": 7, "Eating Out": 8, "Gas": 9, "Mortgage": 10, "Misc": 11}

    df['Category'] = df['Category'].map(category_dic)

    if df.isna().any().any():
        print("Warning: There are NaN values in the data after encoding.")
        print(df[df.isna().any(axis=1)])
    #print("After Encoding:\n", df.head())
    return df


def read_file(filename):
    df = pd.read_csv(filename)

    #print("Raw Data:\n", df.head())

    df = encode_data(df)
    
    #print("Encoded Data:\n", df.head())
    return df

def train_linear_regression(data, alpha = 0.01, iterations = 20000):
    category_models = {}

    for category in data['Category'].unique():
        print(f"Training model for category: {category}")
        category_data = data[data['Category'] == category]

        X,y = prep_data(category_data)

        theta = np.zeros(X.shape[1])

        theta, _ = gradient_descent(X, y, theta, alpha, iterations)
        category_models[category] = theta

    return category_models



    '''
    X, y = prep_data(data)

    #Initialize theta to zero
    theta = np.zeros(X.shape[1])

    #finds best theta
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

    return theta, cost_history
    '''
def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]
    return X.dot(theta)


def predict_next(df, category_models):
    last_month = df['Month'].iloc[-1]
    
    next_month = (last_month % 12) + 1
    # print(last_month) #Check to see if Last Month and Next Month works
    # print(next_month)
    
    predictions = []

    for category, theta in category_models.items():
        X = np.array([[1, next_month]])
        prediction = X.dot(theta)
        predictions.append((category, prediction[0]))

    return predictions

def main():
    data = read_file('/kaggle/input/budget/Budget.csv')

    category_models = train_linear_regression(data)
    print("Trained theta:", category_models)

    predictions = predict_next(data, category_models)

    #X_test = np.array([[1, 1], [1, 2], [1, 3]])  
    #y_test = np.array([1, 2, 3])
    #theta_test = np.array([0.5, 0.5])  

    #cost = compute_cost(X_test, y_test, theta_test)
    #print("Test Cost:", cost)

    print("Predictions for Next Month's Expenses:")
    for category, prediction in predictions:
        print(f"Category {category}: Predicted Expense = {prediction: .2f}")

if __name__ == "__main__":
    main()