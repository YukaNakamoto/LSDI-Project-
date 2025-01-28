import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact

from src.dataset import adjust_set_sizes

def get_config_widgets():
    # standardize_check = widgets.Checkbox(value=False, description='Standardize Data')
    price_interval_ex_outliers_slider = widgets.IntRangeSlider(
        value=[-74.44, 222.01],  # Default range [min, max]
        min=-400,           # Minimum value
        max=900,         # Maximum value
        step=1,          # Step size
        description='Energy Price Interval',
        continuous_update=False  # Update only when sliding stops
    )

    # Create interactive widget
    interact(adjust_set_sizes, 
            train=widgets.FloatSlider(min=0, max=0.9, step=0.01, value=0.8, description='Train Set Size'), 
            val=widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.05, description='Evaluation Set Size'),
            test=widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.15, description='Test Set Size'));

    display(price_interval_ex_outliers_slider)
    
    return price_interval_ex_outliers_slider


def select_features():
    # Feature list
    FEATURES = [
        "hour",
        "dayofyear",
        "dayofweek",
        "is_public_holiday",
        "ma_3_hours",
        "ma_6_hours",
        "ma_1_days",
        "ma_3_days",
        "ma_7_days",
        "Biomass",
        "Hard Coal",
        "Hydro",
        "Lignite",
        "Natural Gas",
        "Nuclear",
        "Other",
        "Pumped storage generation",
        "Solar",
        "Wind offshore",
        "Wind onshore",
        "temperature_2m",
        "precipitation",
        "wind_speed_100m",
        "irradiance"
    ]

    # Dictionary to store the checkbox states
    feature_checkboxes = {feature: widgets.Checkbox(value=True, description=feature) for feature in FEATURES}

    # Function to display checkboxes
    def display_feature_checkboxes():
        checkbox_layout = widgets.VBox(list(feature_checkboxes.values()))
        display(checkbox_layout)

    # Function to get selected features
    def get_selected_features(button):
        selected_features = [feature for feature, checkbox in feature_checkboxes.items() if checkbox.value]
        return selected_features

    # Display checkboxes
    display_feature_checkboxes()

    # Button to confirm selection
    confirm_button = widgets.Button(description="Confirm Selection")
    display(confirm_button)
    return confirm_button.on_click(get_selected_features)

