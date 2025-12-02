from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LRModel:
    def __init__(self, country: str) -> None:
        self.country = country.lower()
        self.guinea_x = None
        self.guinea_y = None

    def load_data(self) -> None:
        """ Function for loading data from the .dat files. """
        try:
            filename = f"ebola_cases_{self.country}.dat"
            df = pd.read_csv(filename, sep="\t")
            self.guinea_x = df["Days"].values.reshape(-1, 1)
            daily_cases= df["NumOutbreaks"].values
            self.guinea_y = np.cumsum(daily_cases).reshape(-1, 1) 
        except FileNotFoundError:
            raise FileNotFoundError("File was not found!")

    def plot(self) -> None:
        """ Function that uses SKLearn to train the model and then plots it with Matplotlib. """

        assert self.guinea_x is not None and self.guinea_y is not None, "You need to call load_data() before calling plot()" 

        model = LinearRegression()
        model.fit(self.guinea_x, self.guinea_y)

        y_pred = model.predict(self.guinea_x)

        plt.plot(self.guinea_x, self.guinea_y, label='Real data')
        plt.plot(self.guinea_x, y_pred, label='Fitted line')
        plt.xlabel('Days')
        plt.ylabel('Cummulative number of outbreaks')
        plt.legend()
        plt.show()

guinea = LRModel("guinea")
guinea.load_data()
guinea.plot()
