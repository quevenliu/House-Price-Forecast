import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GBFGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, K=3):
        self.K = K
        self.breakpoints = None  # Stores breakpoints for x and y coordinates
        self.mu = dict()         # Stores mu values for each i, j
        self.s = dict()          # Stores std deviation values for each i, j
        self.GBF_columns = []    # List to store names of GBF columns

    def generateBreakPoints(self, x, y):
        """Generate breakpoints for x and y coordinates."""
        x_breakpoints = np.linspace(x.min(), x.max(), self.K + 1)
        y_breakpoints = np.linspace(y.min(), y.max(), self.K + 1)
        return x_breakpoints, y_breakpoints

    def fit(self, X, y=None):
        """Fit the GBF generator on the training data by calculating breakpoints, mu, and s values."""
        x = X['橫坐標']
        y = X['縱坐標']

        # Calculate breakpoints for x and y coordinates
        self.breakpoints = self.generateBreakPoints(x, y)

        # Calculate mu and s values within each bin defined by the breakpoints
        for i in range(self.K):
            for j in range(self.K):
                # Determine the mask for x and y based on breakpoints
                if i == self.K - 1:
                    x_mask = (x >= self.breakpoints[0][i])
                else:
                    x_mask = (x >= self.breakpoints[0][i]) & (
                        x < self.breakpoints[0][i + 1])

                if j == self.K - 1:
                    y_mask = (y >= self.breakpoints[1][j])
                else:
                    y_mask = (y >= self.breakpoints[1][j]) & (
                        y < self.breakpoints[1][j + 1])

                mask = x_mask & y_mask

                # If there are enough data points in the bin, calculate mean and std
                if mask.sum() >= 20:
                    mu_x = x[mask].mean()
                    mu_y = y[mask].mean()
                    s_x = x[mask].std()
                    s_y = y[mask].std()

                    # Store the mu and std values
                    self.mu[(i, j)] = (mu_x, mu_y)
                    self.s[(i, j)] = (s_x, s_y)

                    # Define the name for the GBF column and store it
                    gbf_col_name = f'GBF_{i}_{j}'
                    self.GBF_columns.append(gbf_col_name)

        return self

    def transform(self, X):
        """Transform data by generating GBF columns using precomputed mu and s values."""
        for (i, j), (mu_x, mu_y) in self.mu.items():
            # Retrieve standard deviations for this bin
            s_x, s_y = self.s[(i, j)]

            # Calculate the GBF for this bin using the stored mu and s values
            gbf_col_name = f'GBF_{i}_{j}'
            X[gbf_col_name] = self.GBF(
                X['橫坐標'], X['縱坐標'], mu_x, mu_y, s_x, s_y)

        # Ensure all columns with NaNs are filled, which may arise if no data points in some regions
        X = X.fillna(0)

        # Return only the GBF columns generated
        return X[self.GBF_columns]

    def GBF(self, x, y, mu_x, mu_y, s_x, s_y):
        """Calculate the Generalized Bivariate Feature value for each point."""
        # Example GBF calculation
        return np.exp(-((x - mu_x)**2 / (2 * s_x**2) + (y - mu_y)**2 / (2 * s_y**2)))
