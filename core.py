# Main file containing all classes and functions 

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Iterable
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


from enum import Enum

## TASK 2
class TrainingMethod(Enum):
    Linear = "linear"
    Polynomial = "polynomial"

@dataclass
class Country:
    name: str
    x: ArrayLike | None = None
    y: ArrayLike | None = None
    y_pred: ArrayLike | None = None
    poly: PolynomialFeatures | None = None
    model: LinearRegression | None = None
    
    def __post_init__(self) -> None:
        try:
            filename = f"ebola_cases_{self.name.replace(' ', '_').lower()}.dat"
            df = pd.read_csv(filename, sep="\t")
            self.x = df["Days"].values.astype("float32").reshape(-1, 1)
            self.y = df["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
        except FileNotFoundError:
            raise FileNotFoundError("File was not found!")
    
    def train(self, training_method: TrainingMethod):
        assert self.x is not None and self.y is not None, "Failed to load x and y values" 

        if training_method == TrainingMethod.Linear:
            self.model = LinearRegression()
            self.model.fit(self.x, self.y)

            self.y_pred = self.model.predict(self.x)
            
        if training_method == TrainingMethod.Polynomial:
            self.poly = PolynomialFeatures(degree=3, include_bias=False)
            poly_features = self.poly.fit_transform(self.x)

            self.model = LinearRegression()
            self.model.fit(poly_features, self.y)

            self.y_pred = self.model.predict(poly_features)

def plot_countries(countries):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.flatten()
    for i, country in enumerate(countries):
        ax = axes[i]
        ax.scatter(country.x, country.y, marker='o', linestyle='-', label="Real data")
        ax.set_title(country.name)
        ax.set_xlabel("Days")
        ax.set_ylabel("Number of outbreaks")
        
        if country.y_pred is not None:
            ax.plot(country.x, country.y_pred, label="Fitted data", color="orange")
    
    fig.suptitle("Ebola Outbreaks", fontsize=16, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(pad=2.0)
    
    return fig, axes