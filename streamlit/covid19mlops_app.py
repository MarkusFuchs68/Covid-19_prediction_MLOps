# Import Streamlit and other required libraries
import os  # For file/folder handling

import content as content  # Custom module for UI content

import streamlit as st  # Web app UI with Streamlit

# from api_client import (login, register_model, list_models, get_model, predict)

# Define the class names (4-class and 2-class variants) in the right index order of the models
classes_4 = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
classes_2 = ["Normal", "COVID"]
MODEL_FOLDER = os.path.join(
    ".", "models"
)  # Set the folder path where models will be saved


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

elif page == "About":

    st.markdown(content.about)
"""
elif page == 'Model selection':  # If the selected page is 'Model selection'

    # Always offer the download or update of the models
    if st.button('Download/update models'):  # Button to allow the user to download or update the model files
        # We download the models from our Google Drive
        st.write('Downloading models from Google Drive...')  # Display a message to the user
        st.write('Please wait...')
        download_models()  # Call the function to download models
        st.write('Models downloaded successfully. Refreshing page...')  # Notify user and refresh Streamlit
        st.rerun()

    # Check if the models folder exists
    if not os.path.isdir(MODEL_FOLDER):
        st.error('No models available. Please download models first.')
        st.stop()  # Stop the app here

    # Load the models folder
    model_file_list = os.listdir(MODEL_FOLDER)  # List all files inside the model folder
    model_names = [model_filename.split('.')[0] for model_filename in model_file_list if model_filename.endswith('.keras')]  # Only keep model filenames ending with ".keras" and remove extension
    model_names = sorted(model_names)  # Sort the model names alphabetically

    # Check if we have models in the folder
    if len(model_names) == 0:  # If no models found, stop the app
        st.error('No models available. Please download models first.')
        st.stop()

    # Let the user select one or preset it with the last selected one
    selected_model_name = st.selectbox('Select a model for prediction:',
                                       model_names,
                                       key='model_name')
    if selected_model_name is None:  # If nothing is selected, stop the app
        st.stop()

    # Load the selected model
    model = ts.keras.models.load_model(os.path.join(MODEL_FOLDER, selected_model_name + '.keras'))
    st.write('Model loaded successfully.')
    st.write('Model summary:')
    df = pred.model_summary_to_df(model)
    st.dataframe(df, use_container_width=False)

    # Save it in streamlit session state
    classes = classes_2 if '2-classes' in selected_model_name else classes_4
    st.session_state['model'] = model
    st.session_state['classes'] = classes
    st.session_state['selected_model'] = selected_model_name

    # Always offer the download or update of the models
    if st.button('Download/update models'): # Button to allow the user to download or update the model files
        # We download the models from our Google Drive
        st.write('Downloading models from Google Drive...') # Display a message to the user
        st.write('Please wait...')
        download_models() # Call the function to download models
        st.write('Models downloaded successfully. Refreshing page...') # Notify user and refresh Streamlit
        st.rerun()

    # Check if the models folder exists
    if not os.path.isdir(MODEL_FOLDER):
        st.error('No models available. Please download models first.')
        st.stop() # Stop the app here

    # Load the models folder
    model_file_list = os.listdir(MODEL_FOLDER) # List all files inside the model folder
    model_names = [model_filename.split('.')[0] for model_filename in model_file_list if model_filename.endswith('.keras')] # Only keep model filenames ending with ".keras" and remove extension
    model_names = sorted(model_names) # Sort the model names alphabetically

    # Check if we have models in the folder
    if len(model_names) == 0:  # If no models found, stop the app
        st.error('No models available. Please download models first.')
        st.stop()

    # Let the user select one or preset it with the last selected one
    selected_model_name = st.selectbox('Select a model for prediction:',
                                       model_names,
                                       key='model_name')
    if selected_model_name is None: # If nothing is selected, stop the app
        st.stop()

    # Load the selected model
    model = ts.keras.models.load_model(os.path.join(MODEL_FOLDER, selected_model_name + '.keras'))
    st.write('Model loaded successfully.')
    st.write('Model summary:')
    df = pred.model_summary_to_df(model)
    st.dataframe(df, use_container_width=False)

    # Save it in streamlit session state
    classes = classes_2 if '2-classes' in selected_model_name else classes_4
    st.session_state['model'] = model
    st.session_state['classes'] = classes
    st.session_state['selected_model'] = selected_model_name

elif page == 'End User': # If the user navigates to the "Prediction" page
    # First check if we have models loaded
    if ('model' not in st.session_state): # Check if a model is loaded before making predictions
        st.error('No model loaded. Please select a model first.')
        st.stop() # Stop execution if no model is loaded

    model_name = st.session_state['selected_model'] # Load model and class list from Streamlit's session state
    model = st.session_state['model']
    classes = st.session_state['classes']

    # Let the user choose between file upload or URL input
    # Initialize variables
    image = None # To store the loaded image
    loading_type = 0 # 0 = nothing yet, 1 = from URL, 2 = from file upload
    st.markdown(content.prediction_note) # Display a helpful prediction note (markdown block from content file)
    url_input, divider, file_input = st.columns([3, 1, 3]) # Create three columns for layout: one for URL input, one as divider, one for file upload

    with divider: # Write the "or" divider between URL and file input
        st.markdown(content.prediction_or, unsafe_allow_html=True)

    with url_input: # If user uses the URL input field
        st.write('Enter a URL of an X-ray image for prediction:')
        # Text field where the user can paste an image URL (with a default example link)
        image_url = st.text_input('Image URL:', value='https://content.ca.healthwise.net/resources/14.1/en-ca/media/medical/hw/h9991297_001.jpg')
        if image_url and image_url != '': # If a URL is given
            image = pred.load_image_from_url(image_url) # Load image from the web
            st.write(f'You entered: {image_url}') # Show the entered URL
            loading_type = 1 # Mark that image came from URL

    with file_input: # Section to upload an image file instead of a URL
        st.write('Upload an X-ray image for prediction:')
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png']) # User uploads a file (only images: .jpg, .jpeg, .png allowed)
        if uploaded_file:
            image = pred.load_image_from_file(uploaded_file) # If an image file was uploaded, load it
            st.write(f'You uploaded: {uploaded_file.name}') # Confirm upload
            loading_type = 2 # Flag that the image came from file

    # Show the original image
    if image is None: # If no image was loaded from either method, stop the app
        st.stop()

    # Predict using the model
    # Show which model will be used and where the image came from
    if loading_type == 1:
        st.write('Predicting the following image from URL with model:', model_name)
    elif loading_type == 2:
        st.write('Predicting the following image from file with model:', model_name)

    # Show the loaded image
    st.image(image, width=300) # Show the image on screen (resized width to 300px for clarity)

    # Prepare the image for prediction in the selected model
    img_prepared = pred.prepare_image_for_model(image, model_name, model)

    # Predict the prepared image and show the result as table
    pred_df = pred.predict_image(img_prepared, model_name, model, classes) # This returns a table with prediction probabilities
    st.table(pred_df) # Show the prediction table, while maintaining proper column widths by using st.table()

    # Optionally show a Grad-CAM
    st.write('Check the following checkbox to show a Grad-CAM of the prediction:') # Optional: checkbox to show Grad-CAM visualization
    show_gradcam = st.checkbox('Show Grad-CAM')
    if show_gradcam:
        pred.show_feature_maps(img_prepared, model_name, model) # Show feature maps using Grad-CAM
"""
