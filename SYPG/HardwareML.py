import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import *

def output():       
    data = pd.read_csv("computer_hardware_withformula.csv", sep=",") #read in all data from database and exclude the commas

    #predicting Estimated relative performance
    predict = "prp"

    #I chose attributes which are integers and most relevant to improve accuracy
    data = data[["myct","mmin", "mmax","cach", "chmin","chmax","prp"]]
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

    #set up two arrays, one is array defined all of the attributes, and another is the label
    x = np.array(data.drop([predict], axis=1))
    y =np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    #takes in all of the attributes/labels and split into four different arrays (x_train, x_test, y_train, y_test)
    #0.1 so that it splits up 10% of the data into test samples, and let model use 90% information (training dataset) to become good at predicting


    #Train model multiple times to achieve high accuracy
    best = 0
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        linear = linear_model.LinearRegression()
        #linear regression is used as there should be a linear relationship, and the datatypes of attributes in the dataset is integers

        linear.fit(x_train, y_train) #creates the best fit line
        acc = linear.score(x_test, y_test)
        print("Accuracy: " + str(acc)) #prints accuracy for each instance from dataset

        if acc > best:
            best = acc
            with open("hardware_performance.pickle", "wb") as f:
                pickle.dump(linear, f)

    #Loads the linear model
    pickle_in = open("hardware_performance.pickle", "rb")
    linear = pickle.load(pickle_in)

    #printing coefficient and intercept from the best fit line for all attributes used above
    print("------------------------")
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    print("------------------------")

    # prints a list of all predictions as well as the input data from the dataset
    predicted= linear.predict(x_test)
    for x in range(len(predicted)):
        print(predicted[x], x_test[x], y_test[x])
            
def plot(attr):   #investigate the significance of certain attributes affect on Computer performance(PRP)
    plot = str(attr)
    data = pd.read_csv("machine.data", sep=",") #read in all data from database and exclude the commas

    #predicting Estimated relative performance
    predict = "PRP"

    #I chose attributes which are integers and most relevant to improve accuracy
    data = data[["MYCT","MMIN", "MMAX","CACH", "CHMIN","CHMAX","PRP"]]
    

    #set up two arrays, one is array defined all of the attributes, and another is the label
    x = np.array(data.drop([predict], axis=1))
    y =np.array(data[predict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    #takes in all of the attributes/labels and split into four different arrays (x_train, x_test, y_train, y_test)
    #0.1 so that it splits up 10% of the data into test samples, and let model use 90% information (training dataset) to become good at predicting


    plt.scatter(data[plot], data["PRP"])
    plt.legend(loc=4)
    plt.xlabel(plot)
    plt.ylabel("Performance")
    #print(np.interp(20000, x,y))
    plt.show()



# create the tkinter window
root = tk.Tk()
root.title("GUI")                      

# ===== header frame
header_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=2)
header_frame.pack(padx=10,pady=20)
headerLabel = tk.Label(header_frame,text='Supervised Machine Learning Model to predict Computer Performance', bg="white", font=('Helvetica', 18, 'bold'))        
headerLabel.grid()

headerLabel.grid()

# ===== input frame
input_frame = tk.Frame(root)
input_frame.pack()

txtLabel = tk.Label(input_frame, text="Press button to train model using the dataset ", font=('Arial',11,'underline'))
txtLabel.grid(row=1,column=0)

txtLabel2 = tk.Label(input_frame, text="Dataset used: Computer Hardware Data Set from UCI Machine Learning Repository", font=('Arial',7))
txtLabel2.grid(row=0,column=0)

showButton = tk.Button(input_frame, text="           Train          ", bg="seagreen", fg="white", command=output)
showButton.grid(row=1, column=1, padx=5, pady=20)

# ===== input frame
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

#another label and say the dataset using
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

# === output frame
output_frame = tk.Frame(root)
output_frame.pack()

#message 
outLabel = tk.Label(output_frame, text='', font=11)
outLabel.grid(row=0, column=1)

#message 
outLabel2 = tk.Label(output_frame, text='', font=11)
outLabel2.grid(row=1, column=1)

root.mainloop()