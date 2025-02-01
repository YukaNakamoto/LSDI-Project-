import ipywidgets as widgets
from IPython.display import display
from datetime import date

def get_config_widgets():
    # IntRangeSlider for Energy Price Interval
    price_interval_ex_outliers_slider = widgets.IntRangeSlider(
        value=[-500, 900],  # Default range [min, max]
        min=-500,                # Minimum value
        max=900,                 # Maximum value
        step=1,                  # Step size
        description='Energy Price Interval',
        continuous_update=False  # Update only when sliding stops
    )

    # DatePicker for Prediction Date
    prediction_date_picker = widgets.DatePicker(
        value=date.today(),
        description='Prediction Date',
        disabled=False
    )

    # Create interactive widgets for adjusting set sizes
    eval_size = widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.05, description='Evaluation Set Size')

    normalize = widgets.Checkbox(
        value=False, description="Normalize"
    )

    # Display the widgets
    display(price_interval_ex_outliers_slider, prediction_date_picker, eval_size, normalize)
    
    return price_interval_ex_outliers_slider, prediction_date_picker, eval_size, normalize


def select_features():
    # Feature list
    FEATURES = [
        "hour",
        "dayofyear",
        "dayofweek",
        "ma_2_hours",
        "ma_3_hours",
        "ma_4_hours",
        "ma_5_hours",
        "ma_6_hours",
        "ma_7_hours",
        "ma_8_hours",
        "ma_9_hours",
        "ma_10_hours",
        "ma_11_hours",
        "ma_12_hours",
        "ma_13_hours",
        "ma_14_hours",
        "ma_15_hours",
        "ma_16_hours",
        "ma_17_hours",
        "ma_18_hours",
        "ma_19_hours",
        "ma_20_hours",
        "ma_21_hours",
        "ma_22_hours",
        "ma_23_hours",
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
        "ma_3_hours_pumped_storage_generation",
        "ma_6_hours_pumped_storage_generation",
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
