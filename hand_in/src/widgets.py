import ipywidgets as widgets
from IPython.display import display
from datetime import date


def get_config_widgets():
    # IntRangeSlider for Energy Price Interval
    price_interval_ex_outliers_slider = widgets.IntRangeSlider(
        value=[-20, 400],  # Default range [min, max]
        min=-500,  # Minimum value
        max=900,  # Maximum value
        step=1,  # Step size
        description="Energy Price Interval",
        continuous_update=False,  # Update only when sliding stops
    )

    # Create interactive widgets for adjusting set sizes
    eval_size = widgets.FloatSlider(
        min=0, max=0.2, step=0.01, value=0.05, description="Evaluation Set Size"
    )

    normalize = widgets.Checkbox(value=False, description="Normalize")

    # Display the widgets
    display(price_interval_ex_outliers_slider, eval_size, normalize)

    return (
        price_interval_ex_outliers_slider,
        eval_size,
        normalize,
    )


def select_features():
    # Feature list
    FEATURES = [
        "hour",
        "dayofyear",
        "dayofweek",
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
        "direct_radiation",
    ]

    for i in range(2, 24):
        FEATURES.append(f'ma_{i}_hours')
        FEATURES.append(f'ma_{i}_hours_pumped_storage_generation')

    for i in range(1, 15):
        FEATURES.append(f'ma_{i}_days')
        FEATURES.append(f'ma_{i}_days_pumped_storage_generation')

    DEFAULT_FEATURES = [
        "hour",
        "dayofyear",
        "dayofweek",
        "Hydro",
        "Pumped storage generation",
        "ma_3_hours_pumped_storage_generation",
        "ma_6_hours_pumped_storage_generation",
        "Solar",
        "Wind offshore",
        "Wind onshore",
        "temperature_2m",
        "precipitation",
        "wind_speed_100m",
        "direct_radiation",
    ]

    # Create checkboxes for all features.
    feature_checkboxes = {
        feature: widgets.Checkbox(
            value=(feature in DEFAULT_FEATURES.copy()), description=feature
        )
        for feature in FEATURES
    }

    # Organize checkboxes into three columns using a GridBox.
    columns = 3
    checkbox_items = list(feature_checkboxes.values())
    grid = widgets.GridBox(
        checkbox_items,
        layout=widgets.Layout(grid_template_columns=f"repeat({columns}, auto)"),
    )

    # Display the grid of checkboxes.
    display(grid)

    # Create an output widget to display the selected features.
    output = widgets.Output()
    display(output)

    # Button to confirm the selection.
    confirm_button = widgets.Button(description="Confirm Selection")
    display(confirm_button)

    # Variable to store the selected features.
    selected_features = []

    # Function to update and display the selected features.
    def get_selected_features(_):
        nonlocal selected_features
        selected_features = [
            feature
            for feature, checkbox in feature_checkboxes.items()
            if checkbox.value
        ]
        with output:
            output.clear_output()  # Clear any previous output.
            print("Selected Features:", selected_features)

    # Link the button click event to the function.
    confirm_button.on_click(get_selected_features)

    # Return a lambda function that retrieves the selected features,
    # and also return the default features.
    return lambda: selected_features, DEFAULT_FEATURES
