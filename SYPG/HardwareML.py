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

# 04/01/25
# changed output function to train
# added predict function
# fixed training output
# replaced bar with heatmap

import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
import seaborn as sns
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

# Read dataset
def load_data():
    return pd.read_csv("computer_hardware_withformula.csv", sep=",")[["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]]
    '''
    Relevant attributes for predictions
    MYCT: machine cycle time in nanoseconds (integer) - The time it takes for the CPU to execute a single machine cycle, measured in nanoseconds (ns). 
    Lower MYCT = Faster CPU Performance.
    
    MMIN: minimum main memory in kilobytes (integer) - The minimum amount of RAM (random access memory) the system can support, measured in kilobytes (kB).
    Higher MMIN = Better baseline performance.
    
    MMAX: maximum main memory in kilobytes (integer) - The maximum amount of RAM the system can support, measured in kilobytes (kB).
    Higher MMAX = More multitasking capability and performance.
    
    CACH: cache memory in kilobytes (integer) - A small, high-speed memory located inside or near the CPU that stores frequently accessed data for quick retrieval, measured in kilobytes (kB).
    Larger Cache = Faster data retrieval = Better system speed.
    
    CHMIN: minimum channels in units (integer) - The minimum number of data channels available for communication between the CPU and memory/storage, measured in units. More channels allow faster data transfer.
    More channels = Faster data transfer.

    CHMAX: maximum channels in units (integer) - The maximum number of channels available for data transfer, measured in units.
    Higher CHMAX = More simultaneous device connections = Improved system efficiency.

    PRP: published relative performance (integer) - A performance score published in benchmark studies, showing how well the computer performs based on its hardware specifications.
    
    ERP: estimated relative performance from the original article (integer) - The estimated performance value from the original research article, calculated based on a model or equation.
    '''

# Training Output
def train():       
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

    # Layout frames
    main_frame = tk.Frame(root)
    main_frame.pack(pady=10)

    accuracy_frame = tk.Frame(main_frame)
    accuracy_frame.grid(row=0, column=0, padx=10)

    prediction_frame = tk.Frame(main_frame)
    prediction_frame.grid(row=0, column=1, padx=10)

    coefficient_frame = tk.Frame(root)
    coefficient_frame.pack(pady=10)

    # Train the model
    print("\nTrain Results:")
    best = 0 # Tracks the best accuracy score

    # Accuracy label
    tk.Label(accuracy_frame, text="Accuracy:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky="w")

    best_x_test, best_y_test = None, None  # Store best test data

    # Model will be trained 20 times to improve accuracy
    for i in range(20):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=20)

        # Use linear regression and adjust the training data
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train) # Create the best fit line

        # Compute accuracy score
        acc = linear.score(x_test, y_test)

        # Display accuracy in the window
        tk.Label(accuracy_frame, text=f"{acc:.4f}", font=('Arial', 11)).grid(row=i+1, column=0, sticky="we")

        print(f"{i+1}. Accuracy: " + str(acc)) # Print accuracy score for each iteration / TERMINAL

        # Model is saved using pickle, when new accuracy is best
        if acc > best:
            best = acc
            best_x_test, best_y_test = x_test, y_test  # Store best test set

            with open("hardware_performance.pickle", "wb") as f:
                pickle.dump(linear, f)

    # Load best saved model / linear regression model
    with open("hardware_performance.pickle", "rb") as f:
        linear = pickle.load(f)

    # Display the prediction of the performance on x_test and compare it to the y_test / TERMINAL
    print("\nPrediction:")

    tk.Label(prediction_frame, text="Predictions:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky="we")

    predicted = linear.predict(best_x_test)

    # Display prediction
    for i in range(len(predicted)):
        # Display prediction in the window
        tk.Label(prediction_frame, text=f"Predicted: {predicted[i]:.2f}\t|\tActual: {best_y_test[i]}",
                 font=('Arial', 11)).grid(row=i+1, column=0, sticky="w")

        print(f"{i+1}. Predicted: {predicted[i]:.2f}\t|\tActual: {best_y_test[i]}") # TERMINAL

    # Display linear coefficient and intercept in the window
    tk.Label(coefficient_frame, text="Coefficient:", font=('Arial', 11, 'bold')).pack(anchor="center")
    tk.Label(coefficient_frame, text=str(linear.coef_), font=('Arial', 11)).pack(anchor="center")

    tk.Label(coefficient_frame, text="Intercept:", font=('Arial', 11, 'bold')).pack(anchor="center")
    tk.Label(coefficient_frame, text=str(linear.intercept_), font=('Arial', 11)).pack(anchor="center")

    # TERMINAL
    print('\nCoefficient:\n', linear.coef_) # Coefficient values (how much each feature affects PRP)
    print('\nIntercept:\n', linear.intercept_) # Intercept (regression line starts)

# User input prediction
def predict():
    try:
        # Ensure all inputs are valid integers
        user_data = []
        for entry in entries:
            value = entry.get().strip()
            if not value.isdigit():
                messagebox.showerror("Input Error", "Please enter valid integer values for all fields.")
                return
            user_data.append(int(value))

        if len(user_data) != 6:  # Ensure we have 6 inputs
            messagebox.showerror("Input Error", "Please enter values for all 6 fields.")
            return

        # Convert user input into a NumPy array and reshape for the model
        user_data = np.array(user_data).reshape(1, -1)

        # Load the trained model
        with open("hardware_performance.pickle", "rb") as f:
            model = pickle.load(f)

        # Make prediction
        predicted_prp = model.predict(user_data)[0]

        # Display prediction 
        pred_window = tk.Toplevel(root)
        pred_window.title("Prediction Result")
        tk.Label(pred_window, text=f"Predicted Performance (PRP): {predicted_prp:.2f}",
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)

        print(f"\nPredicted Performance (PRP): {predicted_prp:.2f}")  # TERMINAL

        # Create frames to hold the PRP score and performance level information
        prp_score = tk.Frame(pred_window)
        prp_score.grid(row=1, column=0, padx=10)

        performance_level = tk.Frame(pred_window)
        performance_level.grid(row=1, column=1, padx=10)

        # Display PRP score range 
        tk.Label(prp_score, text="PRP Score Range:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky="we")
        tk.Label(prp_score, text="1 - 100 \n101 - 400 \n401 - 800 \n801 - 1150+", font=('Arial', 11)).grid(row=1, column=0, sticky="we")

        # Display performance levels
        tk.Label(performance_level, text="Performance Levels:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky="we")
        tk.Label(performance_level, text="Low Performance: Basic computing, slow processing"
                                         "\nModerate Performance: Good for office tasks"
                                         "\nHigh Performance: Handles intensive tasks like gaming, programming"
                                         "\nVery High Performance: Top-tier computing, powerful workstations/servers", 
                 font=('Arial', 11)).grid(row=1, column=0, sticky="we", padx=5)

    except FileNotFoundError:
        messagebox.showerror("Model Error", "Trained model not found. Please train the model first.")

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

    # Histogram visualization
    elif plot_type == "Histogram":
        attr = selected_attr.get()  # Get the selected attribute from the combo box
        plt.hist(data[attr], bins=20, color="orange", alpha=0.7, edgecolor="black")
        plt.xlabel(attr)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {attr} Distribution")

    # Heatmap visualization (Correlation Matrix)
    elif plot_type == "Heatmap":
        # Compute correlation matrix
        correlation_matrix = data.corr()

        # Create a heatmap using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Heatmap of Hardware Attributes vs PRP")
        plt.show()

    # Display plot
    plt.show()

# Tkinter window / GUI
root = tk.Tk()
root.title("6DANCS | SYPG")                       

# Header frame
header_frame = tk.Frame(root, relief=tk.SUNKEN, borderwidth=2)
header_frame.pack(padx=10,pady=20)
headerLabel = tk.Label(header_frame,text='Predicting Computer Performance Using Machine Learning on Hardware Specifications', bg="white", font=('Helvetica', 18, 'bold'))        
headerLabel.grid()

headerLabel.grid()

# Model Training Buttons (Input)
input_frame = tk.Frame(root)
input_frame.pack()

txtLabel = tk.Label(input_frame, text="Press button to train model through linear regression using the dataset ", font=('Arial',11,'underline'))
txtLabel.grid(row=1,column=0)

txtLabel2 = tk.Label(input_frame, text="Dataset used: UCI Machine Learning Repository Computer Hardware Data Set \nhttps://archive.ics.uci.edu/dataset/29/computer+hardware", font=('Arial',7)) # https://archive.ics.uci.edu/dataset/29/computer+hardware
txtLabel2.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

showButton = tk.Button(input_frame, text="           Train          ", bg="purple", fg="white", command=train)
showButton.grid(row=1, column=1, padx=5, pady=20)

# Drop-down menu for selecting visualization
tk.Label(root, text="Select Visualization:", font=("Arial", 12)).pack()
combo = ttk.Combobox(root, values=["Linear Regression", "Scatter Plot", "Histogram", "Heatmap"], state="readonly")
combo.pack()
combo.set("Select Visualization")

# Variable that track selected attribute
selected_attr = tk.StringVar()
selected_attr.set("MMAX")  # Default attribute

# Visualization Buttons (Input)
input_frame2 = tk.Frame(root)
input_frame2.pack()

txtLabel3 = tk.Label(input_frame2, text="Pick attribute below to show its visualization based from the dataset", font=('Arial',11,'underline'))
txtLabel3.pack(side=TOP, pady=10)

# Attribute selection buttons for visualization
for attr in ["MMAX", "MMIN", "CACH", "CHMIN", "CHMAX"]:
    tk.Radiobutton(input_frame2, text=attr, variable=selected_attr, value=attr).pack(side=tk.LEFT, padx=10)

# Button to generate visualization
plot_button = tk.Button(root, text="Generate Visualization", bg="yellow", fg="black", command=plot)
plot_button.pack(pady=10)

# User Input Prediction (Input)
input_frame3 = tk.Frame(root)
input_frame3.pack()

tk.Label(input_frame3, text="To predict computer performance, enter your integer value specs", 
         font=('Arial', 11, 'underline')).grid(row=0, column=0, columnspan=2, pady=5)

labels = ["Machine Cycle Time in ns", "Minimum Main Memory in kB", "Maximum Main Memory in kB", 
          "Cache Memory in kB", "Minimum Channels in Units", "Maximum Channels in Units"]

entries = [] # Store user input

for i, label in enumerate(labels):
    tk.Label(input_frame3, text=label, font=('Arial', 10)).grid(row=i+1, column=0, sticky="w", padx=5, pady=2)
    entry = tk.Entry(input_frame3, width=15)
    entry.grid(row=i+1, column=1, padx=10, pady=2)
    entries.append(entry)

# Predict Button
predictButton = tk.Button(input_frame3, text="        Predict         ", bg="blue", fg="white", command=predict)
predictButton.grid(row=9, column=1, padx=5, pady=20)

# Output frame
output_frame = tk.Frame(root)
output_frame.pack()

# Loop Tkinter
root.mainloop()