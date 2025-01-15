import streamlit as st
import importlib

# Set up the Streamlit page
st.set_page_config(page_title="Football Match Prediction", page_icon="⚽")

# Title of the main page
st.title("Football Match Prediction")

# Navigation options
page = st.selectbox("Select a page:", [
    "Fetch Fixture Data and Coaches for a Specific Date",
    "Build Training Set",
    "Fetch Odds Data",
    "Build Test Set",
    "Final Predictions",
    "Metrics"# Add the new option here
])

# Dictionary to map page names to their file names
page_mapping = {
    "Fetch Fixture Data and Coaches for a Specific Date": "fetch_fixture_data_and_coaches_from_the_API_for_a_specific_date",
    "Build Training Set": "build_train_set",
    "Fetch Odds Data": "Fetch_odds_data_for_each_date",
    "Build Test Set": "build_test_set",
    "Final Predictions": "Final_Predictions",
    "Metrics": "Metrics"# Map the new option to the script
}

# Import and run the corresponding page script
if page in page_mapping:
    page_script = page_mapping[page]

    try:
        # Dynamically import the selected page script
        module = importlib.import_module(f"pages.{page_script}")
        importlib.reload(module)  # Reload the module to ensure a fresh load

        # Run the selected page script
        module.run()
    except ModuleNotFoundError:
        st.error(f"The module {page_script} could not be found.")
    except AttributeError:
        st.error(f"The module {page_script} does not contain a 'run' function.")
