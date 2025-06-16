import content as content  # Custom module for UI content
import requests
from api_client import get_model, list_models, login, predict, register_model

import streamlit as st  # Web app UI with Streamlit

# Define the class names (4-class and 2-class variants) in the right index order of the models
classes_4 = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
classes_2 = ["Normal", "COVID"]


# The streamlit app code:
st.title("Covid-19 MLOps solution")  # Title of the web app

st.sidebar.title("Navigation")  # Sidebar for navigation
pages = [
    "Data Scientest",
    "End User",
    "About",
]  # List of pages user can choose from in the app
page = st.sidebar.radio(
    "Go to:", pages
)  # Create a radio button to navigate between pages


if page == "Data Scientest":  # If user selected the "Data Scientest" page
    # Show introductory text and explanations from content module
    st.markdown(content.ds_general)

    st.markdown("#### AUTHENTICATION")

    # we first must login and save the jwt token in the session state
    username = st.text_input("Username", "user123")  # Input field for username
    password = st.text_input(
        "Password", "pass123", type="password"
    )  # Input field for password, masked for security

    if st.button("Login Data Scientist"):  # Button to trigger login
        try:
            # Attempt to login with provided credentials
            st.session_state["jwt_token_ds"] = login(username, password)
            st.success(
                "Login successful!"
            )  # Show success message if login is successful
        except Exception as e:
            # If login fails, show an error message
            st.error(f"Login failed: {str(e)}")
            st.stop()

    st.markdown("#### MODEL REGISTRATION")

    # We want to register a model, so we need to let the user specify a model file name
    model_filepath = st.text_input(
        "Model file path", "file_exchange/4_50ep_medparam_4xconv2d_dense128.keras"
    )  # Input field for model file path
    model_name = st.text_input(
        "Model name", "Streamlit Demo Model"
    )  # Input field for model name
    classes = st.selectbox(
        "Classes",
        ["4-classes", "2-classes"],  # Dropdown to select class type
        index=0,  # Default index for the dropdown
    )  # Dropdown for selecting class names, defaulting to 4-class model
    if classes == "4-classes":  # If the user selects the 4-class model
        classes = classes_4  # Set the class names to the 4-class variant
    else:  # If the user selects the 2-class model
        classes = classes_2  # Set the class names to the 2-class variant
    experiment_name = st.text_input(
        "Experiment name", "streamlit_demo"
    )  # Input field for experiment name, defaulting to "streamlit_demo"

    if st.button("Register Model"):  # Button to trigger model registration
        try:
            register_model(
                st.session_state.get(
                    "jwt_token_ds"
                ),  # Use the JWT token from session state
                model_filepath,  # Model file path input by the user
                model_name,  # Name of the model to register in MLFlow
                list(classes),  # Class names for the model
                experiment_name,  # Name of the MLFlow experiment
                max_num=40,
            )
            st.success(
                "Model registered successfully!"
            )  # Show success message if registration is successful
        except Exception as e:
            st.error(f"Model registration failed: {str(e)}")
            st.stop()


elif page == "End User":  # If the user navigates to the "End User" page

    st.markdown(content.user_general)

    st.markdown("#### AUTHENTICATION")

    # we first must login and save the jwt token in the session state
    username = st.text_input("Username", "user123")  # Input field for username
    password = st.text_input(
        "Password", "pass123", type="password"
    )  # Input field for password, masked for security

    if st.button("Login End User"):  # Button to trigger login
        try:
            # Attempt to login with provided credentials
            st.session_state["jwt_token_user"] = login(username, password)
            st.success(
                "Login successful!"
            )  # Show success message if login is successful
        except Exception as e:
            # If login fails, show an error message
            st.error(f"Login failed: {str(e)}")
            st.stop()

    st.markdown("#### MODELS LISTING")

    # We want to select a model, so we need to let the user select a model from the list of models
    if st.button("List Models"):  # Button to trigger model listing
        try:
            models = list_models(
                st.session_state.get("jwt_token_user")
            )  # Load models from the API
            st.session_state["models"] = [
                model["name"] for model in models
            ]  # Store the loaded models in session state
            st.success(
                f"Successfully loaded models. {len(st.session_state['models'])} models available."
            )  # Show the number of loaded models
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()  # Stop execution if model loading fails

    # If models are not loaded, show an error message
    if not st.session_state.get("models"):
        st.error("No models available. Please load models first.")
        st.stop()

    st.markdown("#### MODEL CHOICE")

    # If models are loaded, show them in a selectbox
    st.session_state["selected_model"] = st.selectbox(
        "Select a model for prediction:",
        st.session_state["models"],  # Show model names in the dropdown
        index=0,  # Default to the first model in the list
    )  # Dropdown for selecting a model from the loaded models

    if st.button("Load Model Details"):  # Button to load the selected model details
        if (
            "selected_model" not in st.session_state
            or not st.session_state["selected_model"]
        ):
            st.error("Please select a model first.")
            st.stop()

        try:
            # Attempt to load the selected model details using the API client
            st.session_state["model"] = get_model(
                st.session_state.get(
                    "jwt_token_user"
                ),  # Use the JWT token from session state
                st.session_state[
                    "selected_model"
                ],  # Get the selected model name from session state
            )  # Load the selected model using the API client
            st.success(
                f"Model '{st.session_state['selected_model']}' loaded successfully!"
            )  # Show success message if loading is successful
            with st.expander(
                "Model Details", expanded=False
            ):  # Create an expandable section for model details
                st.write(
                    st.session_state["model"]
                )  # Display the model details in a table format
        except Exception as e:
            st.error(f"Failed to load model details: {str(e)}")
            st.stop()  # Stop execution if model loading fails

    st.markdown("#### PREDICTION")

    # First check if we have models loaded
    if (
        "model" not in st.session_state
    ):  # Check if a model is loaded before making predictions
        st.error("No model loaded. Please load a model first.")
        st.stop()  # Stop execution if no model is loaded

    # If a model is loaded, we can proceed with the prediction
    predicting_model = st.session_state["model"][
        "name"
    ]  # Get the name of the loaded model from session state

    # Let the user choose to predict from a URL
    st.write(
        "Demo URL covid:",
        "https://www.statnews.com/wp-content/uploads/2020/07/1-s2.0-S0735675720302746-gr1_lrg.jpg",
    )
    st.write(
        "Demo URL normal:",
        "https://media.sciencephoto.com/image/c0096770/800wm/C0096770-Healthy_lungs,_X-ray.jpg",
    )
    url = st.text_input(
        "Image URL for prediction",
        "https://www.statnews.com/wp-content/uploads/2020/07/1-s2.0-S0735675720302746-gr1_lrg.jpg",
    )

    if st.button("Predict from URL"):  # Button to trigger prediction from URL
        try:
            # First get the image from the URL as raw bytes
            image = requests.get(url).content  # Fetch the image from the URL
            if not image:
                st.error(
                    "Failed to load image from URL. Please check the URL and try again."
                )
                st.stop()
        except Exception as e:
            st.error(f"Failed to load image from URL: {str(e)}")
            st.stop()

        # Show the loaded image
        st.write(
            "Predicting the following image from URL with model:", predicting_model
        )
        st.image(
            image, width=300
        )  # Show the image on screen (resized width to 300px for clarity)

        try:
            # Attempt to make a prediction using the API client
            result = predict(
                st.session_state.get(
                    "jwt_token_user"
                ),  # Use the JWT token from session state
                predicting_model,  # The name of the model to use for prediction
                image,  # The image data to predict on
            )  # Make a prediction using the API client
            st.success(
                "Prediction successful!"
            )  # Show success message if prediction is successful
            st.write("Prediction Result:")  # Display a header for the prediction result
            st.write(result)  # Display the prediction result in a table format
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()


elif page == "About":

    st.markdown(content.about)
