# 03/30/25
# fixed comments for readability
# capitalized legends in the csv for uniformality
# added legend on scatter plot, "Data Points"
# added spacings on each accuracy
# removed machine.data, it is just the same as the csv
# removed unnecessary codes

# 03/31/25
# added window for train results
# added ttk on library
# added drop-down menu for visualizations (linear regression, scatter plot, bar graph, and histogram)
# added separate function for loading dataset to avoid repetitive line of codes
# shorten/compressed the attribute buttons
# added comments for display that will only be shown on terminal
# added selected visualization generate button

# GOAL: ENHANCE GUI, CHECK VISUALIZATION AND ANALYZE THE RESULTS IF THEY ARE ACCURATE OR NOT, FIX TRAINING OUTPUT RESULTS

import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import *
from tkinter import ttk

# Read dataset
def load_data():
    return pd.read_csv("computer_hardware_withformula.csv", sep=",")[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]]
    '''
    Relevant attributes for predictions
    MYCT: machine cycle time in nanoseconds (integer)
    MMIN: minimum main memory in kilobytes (integer)
    MMAX: maximum main memory in kilobytes (integer)
    CACH: cache memory in kilobytes (integer)
    CHMIN: minimum channels in units (integer)
    CHMAX: maximum channels in units (integer)
    PRP: published relative performance (integer)
    ERP: estimated relative performance from the original article (integer)
    '''

# Training Output
def output():       
    data = load_data() # Load dataset

    # Predict estimated published relative performance / TARGET VARIABLE
    predict = "PRP"

    x = np.array(data.drop([predict], axis=1)) # Independent variables (MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP)
    y = np.array(data[predict]) # Dependent variable (PRP)

    # 90% training (x_train, y_train) & 10% testing which is represented by 0.1 (x_test, y_test)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    # Train results window
    root = tk.Tk()
    root.title("Train Results")          

    header_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=2)
    header_frame.pack(padx=10,pady=20)
    headerLabel = tk.Label(header_frame,text='Train Results:', bg="white", font=('Helvetica', 18, 'bold'))        
    headerLabel.grid()

    results_frame = tk.Frame(root)
    results_frame.pack(pady=10)

    # Train the model
    print("\nTrain Results:")
    best = 0 # Tracks the best accuracy score

    # Model will be trained 20 times to improve accuracy
    for _ in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        # Use linear regression and adjust the training data
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train) # Create the best fit line

        # Compute accuracy score
        acc = linear.score(x_test, y_test)

        # Display accuracy in the window
        txtLabel = tk.Label(results_frame, text=f"Accuracy:  " + str(acc), font=('Arial', 11))  
        txtLabel.pack(anchor="w")

        print("Accuracy: " + str(acc)) # Print accuracy score for each iteration / TERMINAL

        # Model is saved using pickle, when new accuracy is best
        if acc > best:
            best = acc
            with open("hardware_performance.pickle", "wb") as f:
                pickle.dump(linear, f)

    # Load best saved model / linear regression model
    pickle_in = open("hardware_performance.pickle", "rb")
    linear = pickle.load(pickle_in)

    # Display linear coefficient and intercept in the window
    txtLabel = tk.Label(results_frame, text=("Coefficient: \n", linear.coef_), font=('Arial', 11))  
    txtLabel.pack(anchor="w")

    txtLabel = tk.Label(results_frame, text=("Intercept: \n", linear.intercept_), font=('Arial', 11))  
    txtLabel.pack(anchor="w")

    # TERMINAL
    print("------------------------")
    print('Coefficient: \n', linear.coef_) # Coefficient values (how much each feature affects PRP)
    print('Intercept: \n', linear.intercept_) # Intercept (regression line starts)
    print("------------------------")

    # Display the prediction of the performance on x_test and compare it to the y_test / TERMINAL
    print("Prediction:")
    predicted = linear.predict(x_test)
    for x in range(len(predicted)):

        # Display Prediction in the window
        txtLabel = tk.Label(results_frame, text=(predicted[x], "\t", x_test[x], "\t", y_test[x]), font=('Arial', 11))  
        txtLabel.pack(anchor="w")

        print(predicted[x], "\t", x_test[x], "\t", y_test[x]) # TERMINAL
            




# Show selected plot visualization
def plot(attr=None):
    data = load_data()
    
    plot_type = combo.get() # Retrieves selected visualization from drop-down menu

    # Linear regression visualization
    if plot_type == "Linear Regression":

        # Training and testing same as output() function
        x = np.array(data.drop(["PRP"], axis=1)) # All column except PRP
        y = np.array(data["PRP"]) # Target variable, PRP
        
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) 
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        plt.scatter(y_test, predictions, color="green", alpha=0.5, label="Predicted vs Actual")
        
        # Regression line (y = x for perfect predictions)
        min_val = min(y_test.min(), predictions.min())  # Minimum value for the line
        max_val = max(y_test.max(), predictions.max())  # Maximum value for the line
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Best Fit")
        
        plt.xlabel("Actual PRP")
        plt.ylabel("Predicted PRP")
        plt.title("Linear Regression Predictions")
        plt.legend()
    
    # Scatter plot visualization
    elif plot_type == "Scatter Plot":
        attr = selected_attr.get() 
        plt.scatter(data[attr], data["PRP"], label="Data Points", color="blue", alpha=0.5) # Plots attribute vs. PRP as a scatter plot using blue dots
        plt.xlabel(attr)
        plt.ylabel("Performance")
        plt.title(f"Scatter Plot of {attr} vs PRP")
        plt.legend()


    # Bar graph visualization
    elif plot_type == "Bar Graph":
        attr = attr if attr else "MMAX"  # Default to MMAX if no attribute is selected
        sorted_data = data.sort_values(by=attr).head(10)  # Show top 10 for better visibility on the graph
        plt.bar(sorted_data[attr], sorted_data["PRP"], color="purple", alpha=0.7)
        plt.xlabel(attr)
        plt.ylabel("Performance")
        plt.title(f"Bar Graph of {attr} vs PRP")

    # Histogram visualization
    elif plot_type == "Histogram":
        plt.hist(data["PRP"], bins=20, color="orange", alpha=0.7, edgecolor="black") # Uses 20 bins for histogram
        plt.xlabel("Performance (PRP)")
        plt.ylabel("Frequency")
        plt.title("Histogram of PRP Distribution")

    # Display plot
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

txtLabel = tk.Label(input_frame, text="Press button to train model through linear regression using the dataset ", font=('Arial',11,'underline'))
txtLabel.grid(row=1,column=0)

txtLabel2 = tk.Label(input_frame, text="Dataset used: Computer Hardware Data Set from UCI Machine Learning Repository", font=('Arial',7))
txtLabel2.grid(row=0,column=0)

showButton = tk.Button(input_frame, text="           Train          ", bg="seagreen", fg="white", command=output)
showButton.grid(row=1, column=1, padx=5, pady=20)





# Drop-down menu for selecting visualization
tk.Label(root, text="Select Visualization:", font=("Arial", 12)).pack()
combo = ttk.Combobox(root, values=["Linear Regression", "Scatter Plot", "Bar Graph", "Histogram"], state="readonly")
combo.pack()
combo.set("Select Visualization")

# Variable that track selected attribute
selected_attr = tk.StringVar()
selected_attr.set("MMAX")  # Default attribute

# Visualization Buttons (Input)
input_frame2 = tk.Frame(root)
input_frame2.pack()

txtLabel3 = tk.Label(input_frame2, text="Pick attribute below to show its visualization", font=('Arial',11,'underline'))
txtLabel3.pack(side=TOP, pady=10)

# Attribute selection buttons for visualization
for attr in ["MMAX", "MMIN", "CACH", "CHMIN", "CHMAX"]:
    tk.Radiobutton(input_frame2, text=attr, variable=selected_attr, value=attr).pack(side=tk.LEFT, padx=10)

# Button to generate visualization
plot_button = tk.Button(root, text="Generate Visualization", bg="seagreen", fg="white", command=plot)
plot_button.pack(pady=10)





# User Input Prediction (Input)
input_frame3 = tk.Frame(root)
input_frame3.pack()

tk.Label(input_frame3, text="To predict computer performance, enter your integer value specs", 
         font=('Arial', 11, 'underline')).grid(row=0, column=0, columnspan=2, pady=5)

labels = ["Minimum Main Memory in kB", "Maximum Main Memory in kB", 
          "Cache Memory in kB", "Minimum Channels in Units", "Maximum Channels in Units"]

entries = []  # Store user input

for i, label in enumerate(labels):
    tk.Label(input_frame3, text=label, font=('Arial', 10)).grid(row=i+1, column=0, sticky="w", padx=5, pady=2)
    entry = tk.Entry(input_frame3, width=15)
    entry.grid(row=i+1, column=1, padx=10, pady=2)
    entries.append(entry)

# Predict Button
predictButton = tk.Button(input_frame3, text="        Predict         ", bg="seagreen", fg="white", command=output)
predictButton.grid(row=9, column=1, padx=5, pady=20)

# Output frame
output_frame = tk.Frame(root)
output_frame.pack()

# Loop Tkinter
root.mainloop()