"""Task 1 and 2"""

from enum import Enum
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


class TrainingMethod(Enum):
    """Enum for training method."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SEQUENTIAL = "sequential"


@dataclass
class Country:
    """Class for country used for training and plotting"""

    name: str
    x: ArrayLike | None = None
    y: ArrayLike | None = None
    y_pred: ArrayLike | None = None
    poly: PolynomialFeatures | None = None
    model: LinearRegression | Sequential | None = None

    def __post_init__(self) -> None:
        try:
            filename = f"ebola_cases_{self.name.replace(' ', '_').lower()}.dat"
            df = pd.read_csv(filename, sep="\t")
            self.x = df["Days"].values.astype("float32").reshape(-1, 1)
            self.y = df["NumOutbreaks"].values.astype("float32").reshape(-1, 1)
        except FileNotFoundError as exc:
            raise FileNotFoundError("File was not found!") from exc

    def train(self, training_method: TrainingMethod):
        """Function for training the country data, with either Linear, Polynomial and Sequential."""

        assert self.x is not None and self.y is not None, (
            "Failed to load x and y values"
        )

        if training_method == TrainingMethod.LINEAR:
            self.model = LinearRegression()
            self.model.fit(self.x, self.y)

            self.y_pred = self.model.predict(self.x)

        if training_method == TrainingMethod.POLYNOMIAL:
            self.y_pred = None

            self.poly = PolynomialFeatures(degree=3, include_bias=False)
            poly_features = self.poly.fit_transform(self.x)

            self.model = LinearRegression()
            self.model.fit(poly_features, self.y)

            self.y_pred = self.model.predict(poly_features)

        if training_method == TrainingMethod.SEQUENTIAL:
            split = int(0.70 * len(self.x))
            x_train, y_train = self.x[:split], self.y[:split]

            mean_x = x_train.mean()
            std_x = x_train.std()

            x_train_norm = (x_train - mean_x) / std_x
            x_norm = (self.x - mean_x) / std_x

            y_mean = y_train.mean()
            y_std = y_train.std()

            y_train_norm = (y_train - y_mean) / y_std

            self.model = Sequential(
                [
                    Dense(
                        64,
                        activation="relu",
                        input_shape=(1,),
                        kernel_regularizer=l2(0.01),
                    ),
                    Dense(64, activation="relu"),
                    Dense(1),
                ]
            )

            self.model.compile(optimizer="adam", loss="mean_squared_error")

            self.model.fit(x_train_norm, y_train_norm, epochs=20, verbose=1)

            y_pred_nn = self.model.predict(x_norm)
            self.y_pred = y_pred_nn * y_std + y_mean


def plot_countries(countries):
    """Function for plotting the countries as subplots."""

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes = axes.flatten()
    for i, country in enumerate(countries):
        ax = axes[i]
        ax.scatter(country.x, country.y, marker="o", linestyle="-", label="Real data")
        ax.set_title(country.name)
        ax.set_xlabel("Days since first outbreak")
        ax.set_ylabel("Number of outbreaks")

        if country.y_pred is not None:
            label = (
                "NN predicted data"
                if isinstance(country.model, Sequential)
                else "Fitted data"
            )
            ax.plot(country.x, country.y_pred, label=label, color="orange")

    fig.suptitle("Ebola Outbreaks", fontsize=16, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(pad=2.0)

    return fig, axes
