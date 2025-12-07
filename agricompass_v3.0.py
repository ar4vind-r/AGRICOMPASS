import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageTk
import csv
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
def train_model():
       warnings.filterwarnings("ignore")
       # Create a random forest classifier
       global clf
       clf = RandomForestClassifier(n_estimators=100,max_depth=10)
       # Load the dataset
       df = pd.read_csv('new_crop_details.csv')
       # Split the dataset into training and testing sets
       X_train, X_test, y_train, y_test = train_test_split(df[['Season','Nmin','Nmax', 'Pmin','Pmax', 'Kmin', 'Kmax','pHmin','pHmax','Water']], df['Crop'], test_size=0.4, random_state=42)
       # Train the classifier
       clf.fit(X_train, y_train)

       # Make predictions on the testing set
       y_pred = clf.predict(X_test)

       # Calculate the accuracy of the predictions
       accuracy = np.mean(y_pred == y_test)
       print(accuracy)
       feature_importances = clf.feature_importances_
       #print(feature_importances)


def drawscreen():
    global panchayat
    global season
    global water
    panchayat = panchayat_var.get()
    selected_water_availability = water_availability_var.get()
    selected_season = season_var.get()
    if selected_water_availability=="High":
        water=3
    elif selected_water_availability=="Medium":
        water=2
    else:
        water=1
    if selected_season=="Kharif":
        season=1
    elif selected_season=="Rabi":
        season=2
    else:
        season=3 
     
    crop_name=predict_crop(panchayat,season,water)
    #Display the selected values
    result_label.config(text=f"Panchayat: {panchayat}\t"
                             f"Water availability: {selected_water_availability}\t"
                             f"Season: {selected_season}\t"
                             f"Suitable Crop: {crop_name}")
    price_predict(crop_name)
    

def predict_crop(panchayat,season,water):
       # To predict the crop for a given soil conditions, you can use the following code:
       f=open('PPPROFILE.csv')       
       reader=csv.reader(f)
       rec=list(reader)
       for i in range(1,len(rec)):
              if rec[i][1]==panchayat:
                  Nmin=float(rec[i][2])
                  Nmax=float(rec[i][3])
                  Pmin=float(rec[i][4])
                  Pmax=float(rec[i][5])
                  Kmin=float(rec[i][6])
                  Kmax=float(rec[i][7])
                  pHmin=float(rec[i][8])
                  pHmax=float(rec[i][9])
           

       # Get the soil conditions
       soil_conditions = [season,Nmin,Nmax, Pmin,Pmax, Kmin, Kmax,pHmin,pHmax,water]


       # Make a prediction
       crop_prediction = clf.predict([soil_conditions])[0]
       # Print the crop prediction
       print('Crop predicted:', crop_prediction)
       #Display the selected values
       soil_label.config(text=f"Nitrogen: {Nmin,Nmax}\t"
                             f"Phosphorus: {Pmin,Pmax}\t"
                             f"Potassium: {Kmin, Kmax}\t"
                             f"pH level: {pHmin,pHmax}\n")

       return crop_prediction


def main_window():
    # Create the main window
    global root,result_label,soil_label
    root = tk.Tk()
    root.title("AgriCompass")

     # Load and resize the logo image
    logo_image = Image.open("LOGO (1).png")
    logo_image = logo_image.resize((100, 100))
    logo_photo = ImageTk.PhotoImage(logo_image)

    # Create and place labels for logo and heading using grid
    heading_frame = ttk.Frame(root)
    heading_frame.grid(row=0, column=0, columnspan=2, pady=20)

    logo_label = ttk.Label(heading_frame, image=logo_photo)
    logo_label.grid(row=0, column=0, padx=10)

    heading_label = ttk.Label(heading_frame, text="AGRICOMPASS", font=('Helvetica', 16, 'bold'))
    heading_label.grid(row=0, column=1)

    # Create and place labels
    tk.Label(root, text="Choose Your Panchayat").grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
    tk.Label(root, text="Select water availability").grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)
    tk.Label(root, text="Harvesting season").grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)
  
    result_label = tk.Label(root, text="", font=('Helvetica', 10, 'bold'))
    result_label.grid(row=5, column=0, columnspan=2, pady=10)
    soil_label = tk.Label(root, text="", font=('Helvetica', 10, 'bold'))
    soil_label.grid(row=6, column=0, columnspan=2, pady=10)
    # Create and configure dropdowns
    f=open('PPPROFILE.csv')
    panchayats=[]
    reader=csv.reader(f)
    rec=list(reader)
    for i in range(1,len(rec)):
        panchayats.append(rec[i][1])
    water_availabilities = ["High", "Medium", "Low"]
    seasons = ["Rabi", "Kharif", "All Season"]

    global panchayat_var
    panchayat_var = tk.StringVar(value="select")
    global water_availability_var
    water_availability_var = tk.StringVar(value="select")
    global season_var
    season_var = tk.StringVar(value="select")

    panchayat_dropdown = ttk.Combobox(root, textvariable=panchayat_var, values=panchayats, state="readonly")
    water_availability_dropdown = ttk.Combobox(root, textvariable=water_availability_var, values=water_availabilities, state="readonly")
    season_dropdown = ttk.Combobox(root, textvariable=season_var, values=seasons, state="readonly")

    panchayat_dropdown.grid(row=1, column=1, padx=10, pady=5)
    water_availability_dropdown.grid(row=2, column=1, padx=10, pady=5)
    season_dropdown.grid(row=3, column=1, padx=10, pady=5)

    button = tk.Button(root, text="അനുയോജ്യമായ വിള", command=drawscreen)
    button.grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()
    
def price_predict(crop_name):
    # Load data from CSV
    df = pd.read_csv("cropprice.csv")

    # Unique crop names in the dataset
    crop_names = df['Crop'].unique()
    crop_name=crop_name.strip().upper()
    # Iterate over crops
    if crop_name in crop_names:
        # Filter data for the specific crop
        crop_data = df[df['Crop'] == crop_name]      


        # Extract relevant columns
        crop_data['Date'] = pd.to_datetime(crop_data['Date'],format='%d-%m-%Y', errors='coerce')
        crop_prices = crop_data['Price'].values.reshape(-1, 1)

        # Normalize prices using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(crop_prices)

        # Prepare data for LSTM
        look_back = 3  # Number of previous time steps to use for prediction
        X, y = [], []

        for i in range(len(prices_scaled) - look_back):
            X.append(prices_scaled[i:(i + look_back), 0])
            y.append(prices_scaled[i + look_back, 0])

        X, y = np.array(X), np.array(y)

        # Reshape input data for LSTM (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(1, look_back)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=1, batch_size=1, verbose=2)

        #Future Dates
        fd=['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31','2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']
        # Predict future prices
        future_dates=pd.DatetimeIndex(fd, dtype='datetime64[ns]', freq='Q')
        future_prices_scaled = []

        for i in range(len(future_dates)):
            input_data = prices_scaled[-look_back:]
            input_data = np.reshape(input_data, (1, 1, look_back))
            predicted_price_scaled = model.predict(input_data)
            future_prices_scaled.append(predicted_price_scaled[0, 0])
            prices_scaled = np.append(prices_scaled, predicted_price_scaled, axis=0)

        # Inverse transform to get original scale
        future_prices = scaler.inverse_transform(np.array(future_prices_scaled).reshape(-1, 1)).flatten()
        print("Future Price Prediction using LSTM for", crop_name)
        print(fd)
        print(future_prices )
        df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y', errors='coerce')
        
        fdf = df[df['Crop'] == crop_name].groupby(df['Date'].dt.to_period("M")).first()

        # Rename the 'Date' column in the filtered DataFrame
        fdf.rename(columns={'Date': 'FDate'}, inplace=True)

        if(1):
            #plt.figure(figsize=(10, 6))
            fig,ax=plt.subplots(figsize=(5, 3))
            ax.plot(fdf['FDate'], fdf['Price'], label=f'Historical Prices for {crop_name}', marker='o')
            ax.plot(future_dates, future_prices, label=f'Predicted Future Prices for {crop_name}', marker='o')
            ax.set_title(f'Future Price Prediction using LSTM for {crop_name}', fontsize=8)
            ax.set_xlabel('Date',fontsize=8)
            ax.set_ylabel('Price',fontsize=8)
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=7, column=0, columnspan=3, pady=10)
    else:
        print("Crop not in list",crop_name)
train_model()
main_window()
