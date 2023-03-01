from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px

def plot_one_sample(features: pd.DataFrame, targets: pd.Series, example_id: int, predictions: Optional[pd.Series] = None):
    """Plot one sample from the dataset.
    
    Args:
        features (pd.DataFrame): The features of the dataset.
        targets (pd.Series): The targets of the dataset.
        example_id (int): The id of the example to plot.
        predictions (Optional[pd.Series], optional): The predictions of the model. Defaults to None.
    """

    feature_columns = features.columns[~features.columns.isin(["pickup_hour", "pickup_location_id"])]
    features_ = features.iloc[example_id]
    targets_ = targets.iloc[example_id]

    # datetime range using min and max hours
    num_hours = len(feature_columns)
    min_hour = features_["pickup_hour"] - pd.Timedelta(hours=num_hours)

    # Create the figure
    fig = px.line(
        title=f"Pickup Hour: {features_['pickup_hour']}, Location_ID: {features_['pickup_location_id']}, Example ID: {example_id}",
        x=pd.date_range(min_hour, features_["pickup_hour"], freq="H"),
        y=np.append(features_[feature_columns].values, targets_),
        labels={"x": "Time", "y": "Value"},
    )
    # Add the target
    fig.add_scatter(
        x=pd.date_range(min_hour, features_["pickup_hour"], freq="H")[-1:],
        y=[targets_],
        mode="markers", 
        marker_size=10,
        name="Actual",
        line=dict(color="red", width=2),
    )
    # Add the prediction if available
    if predictions is not None:
        predictions_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=pd.date_range(min_hour, features_["pickup_hour"], freq="H")[-1:],
            y=[predictions_],
            mode="markers",
            marker_size=10,
            name="Prediction",
            line=dict(color="green", width=2),
        )
    # Show the figure
    return fig
