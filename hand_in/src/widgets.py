import ipywidgets as widgets
from IPython.display import display

def get_config_widgets():
    # IntRangeSlider for Energy Price Interval
    price_interval_ex_outliers_slider = widgets.IntRangeSlider(
        value=[-74.44, 222.01],  # Default range [min, max]
        min=-400,                # Minimum value
        max=900,                 # Maximum value
        step=1,                  # Step size
        description='Energy Price Interval',
        continuous_update=False  # Update only when sliding stops
    )

    # DatePicker for Prediction Date
    prediction_date_picker = widgets.DatePicker(
        description='Prediction Date',
        disabled=False
    )

    # Create interactive widgets for adjusting set sizes
    eval_size = widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.05, description='Evaluation Set Size')

    # Display the widgets
    display(price_interval_ex_outliers_slider, prediction_date_picker, eval_size)
    
    return price_interval_ex_outliers_slider, prediction_date_picker, eval_size


def select_features():
    # Feature list
    FEATURES = [
        "hour",
        "dayofyear",
        "dayofweek",
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
        "direct_radiation"
    ]

    # Dictionary to store the checkbox states
    feature_checkboxes = {feature: widgets.Checkbox(value=True, description=feature) for feature in FEATURES}

    # Display checkboxes
    checkbox_layout = widgets.VBox(list(feature_checkboxes.values()))
    display(checkbox_layout)

    # Create an output widget to display the selected features
    output = widgets.Output()
    display(output)

    # Button to confirm selection
    confirm_button = widgets.Button(description="Confirm Selection")
    display(confirm_button)

    # Variable to store the selected features
    selected_features = []

    # Function to get selected features and update output
    def get_selected_features(_):
        nonlocal selected_features
        selected_features = [feature for feature, checkbox in feature_checkboxes.items() if checkbox.value]
        with output:
            output.clear_output()  # Clear previous output
            print("Selected Features:", selected_features)

    # Link the button click event to the function
    confirm_button.on_click(get_selected_features)

    # Return the selected features variable for further use
    return lambda: selected_features, FEATURES
