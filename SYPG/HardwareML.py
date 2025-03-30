# fixed comments for readability
# capitalized legends in the csv for uniformality
# added legend on scatter plot, "Data Points"
# added spacings on each accuracy
# removed machine.data, it is just the same as the csv
# removed unnecessary codes

# GOAL: ADD MORE VISUALIZATION/MODELS (THROUGH DROP-DOWN? - LINEAR REGRESSION, BAR GRAPH, HISTOGRAM), REVAMP GUI, ANALYZE THE RESULTS IF THEY ARE ACCURATE OR NOT

import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import *

# Training Output
def output():       
    data = pd.read_csv("computer_hardware_withformula.csv", sep=",") # Read dataset

    # Predict estimated published relative performance / TARGET VARIABLE
    predict = "PRP"

    # Relevant attributes for predictions
    data = data[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]]
    '''
    MYCT: machine cycle time in nanoseconds (integer)
    MMIN: minimum main memory in kilobytes (integer)
    MMAX: maximum main memory in kilobytes (integer)
    CACH: cache memory in kilobytes (integer)
    CHMIN: minimum channels in units (integer)
    CHMAX: maximum channels in units (integer)
    PRP: published relative performance (integer)
    ERP: estimated relative performance from the original article (integer)
    '''

    x = np.array(data.drop([predict], axis=1)) # Independent variables (MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP)
    y = np.array(data[predict]) # Dependent variable (PRP)

    # 90% training (x_train, y_train) & 10% testing which is represented by 0.1 (x_test, y_test)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    # Train the model
    print("\nTrain Results:")
    best = 0 # Tracks the best accuracy score

    # Model will be trained 20 time to improve accuracy
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        # Use linear regression and adjust the training data
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train) # Create the best fit line

        # Compute accuracy score
        acc = linear.score(x_test, y_test)
        print("Accuracy: " + str(acc)) # Print score for each iteration

        # Model is saved using pickle, when new accuracy is best
        if acc > best:
            best = acc
            with open("hardware_performance.pickle", "wb") as f:
                pickle.dump(linear, f)

    # Load best saved model / linear regression model
    pickle_in = open("hardware_performance.pickle", "rb")
    linear = pickle.load(pickle_in)

    print("------------------------")
    print('Coefficient: \n', linear.coef_) # Coefficient values (how much each feature affects PRP)
    print('Intercept: \n', linear.intercept_) # Intercept (regression line starts)
    print("------------------------")

    # Display the prediction of the performance on x_test and compare it to the y_test
    predicted = linear.predict(x_test)
    for x in range(len(predicted)):
        print(predicted[x], "\t", x_test[x], "\t", y_test[x])
            
# Scatter Plot Output           
def plot(attr): 
    data = pd.read_csv("computer_hardware_withformula.csv", sep=",") 

    data = data[["MYCT","MMIN", "MMAX","CACH", "CHMIN","CHMAX","PRP"]]

    # X-axis (selected attributes) & Y-axis (PRP)
    plt.scatter(data[attr], data["PRP"], label="Data Points")
    plt.legend(loc=4)
    plt.xlabel(attr)
    plt.ylabel("Performance")
    plt.show()

# Tkinter window / GUI
root = tk.Tk()
root.title("6DANCS | SYPG")                       

# Header frame
header_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=2)
header_frame.pack(padx=10,pady=20)
headerLabel = tk.Label(header_frame,text='Supervised Machine Learning Model to predict Computer Performance', bg="white", font=('Helvetica', 18, 'bold'))        
headerLabel.grid()

headerLabel.grid()

# Model Training Buttons (Input)
input_frame = tk.Frame(root)
input_frame.pack()

txtLabel = tk.Label(input_frame, text="Press button to train model using the dataset ", font=('Arial',11,'underline'))
txtLabel.grid(row=1,column=0)

txtLabel2 = tk.Label(input_frame, text="Dataset used: Computer Hardware Data Set from UCI Machine Learning Repository", font=('Arial',7))
txtLabel2.grid(row=0,column=0)

showButton = tk.Button(input_frame, text="           Train          ", bg="seagreen", fg="white", command=output)
showButton.grid(row=1, column=1, padx=5, pady=20)

# Scatter Plot Buttons (Input)
input_frame2 = tk.Frame(root)
input_frame2.pack()

txtLabel3 = tk.Label(input_frame2, text="Pick attribute below to show its scatterplot relationship", font=('Arial',11,'underline'))
txtLabel3.pack(side=TOP, pady=10)

MMAXButton = tk.Button(input_frame2, text="MMAX", bg="mediumseagreen", fg="white", command=lambda:plot('MMAX'))
MMAXButton.pack(side=LEFT, padx=10)
MMINButton = tk.Button(input_frame2, text="MMIN", bg="mediumseagreen", fg="white", command=lambda:plot('MMIN'))
MMINButton.pack(side=LEFT, padx=10)
CACHButton = tk.Button(input_frame2, text="CACH", bg="mediumseagreen", fg="white", command=lambda:plot('CACH'))
CACHButton.pack(side=LEFT, padx=10)
CHMINButton = tk.Button(input_frame2, text="CHMIN", bg="mediumseagreen", fg="white", command=lambda:plot('CHMIN'))
CHMINButton.pack(side=LEFT, padx=10)
CHMAXButton = tk.Button(input_frame2, text="CHMAX", bg="mediumseagreen", fg="white", command=lambda:plot('CHMAX'))
CHMAXButton.pack(side=LEFT, padx=10, pady=20)

# User Input Prediction (Input)
input_frame3 = tk.Frame(root)
input_frame3.pack()

inLabel2 = tk.Label(input_frame3, text="To predict computer performance, enter your integer value specs", font=('Arial',11,'underline'))
inLabel2.grid(row=3,column=0)
inLabel3 = tk.Label(input_frame3, text="Minimum Main Memory in kB", font=('Arial',10))
inLabel3.grid(row=4,column=0)
inLabel4 = tk.Label(input_frame3, text="Maximum Main Memory in kB", font=('Arial',10))
inLabel4.grid(row=5,column=0)
inLabel5 = tk.Label(input_frame3, text="Cache Memory in kB", font=('Arial',10))
inLabel5.grid(row=6,column=0)
inLabel6 = tk.Label(input_frame3, text="Minimum channels in Units", font=('Arial',10))
inLabel6.grid(row=7,column=0)
inLabel7 = tk.Label(input_frame3, text="Maximum channels in Units", font=('Arial',10))
inLabel7.grid(row=8,column=0)

textEntry2 = tk.Entry(input_frame3,text="",width=15)
textEntry2.grid(row=4,column=1, padx=20)
textEntry3 = tk.Entry(input_frame3,text="",width=15)
textEntry3.grid(row=5,column=1)
textEntry4 = tk.Entry(input_frame3,text="",width=15)
textEntry4.grid(row=6,column=1)
textEntry5 = tk.Entry(input_frame3,text="",width=15)
textEntry5.grid(row=7,column=1)
textEntry6 = tk.Entry(input_frame3,text="",width=15)
textEntry6.grid(row=8,column=1)

predictButton = tk.Button(input_frame3, text="        Predict         ", bg="seagreen", fg="white", command=output)
predictButton.grid(row=9, column=1, padx=5, pady=20)

# Output frame
output_frame = tk.Frame(root)
output_frame.pack()

# Loop Tkinter
root.mainloop()