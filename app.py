import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import tempfile
import os
import shutil
import zipfile
from PIL import Image
import numpy as np
import io
import gdown
import pandas as pd
import time
from folium import plugins



import tensorflow as tf
import matplotlib
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt


# url = 'https://drive.google.com/uc?id=1DBl_LcIC3-a09bgGqRPAsQsLCbl9ZPJX'
if 'button1' not in st.session_state:
    st.session_state.button1=False
if 'zoomed_in' not in st.session_state:
    st.session_state.zoomed_in=True
def callback():
    st.session_state.button1=True
def callback_map():
    st.session_state.button1=False
    st.session_state.india_map = create_map(5)
    st.session_state.zoomed_in=True    

@st.cache_resource(show_spinner = False)
def download_model():
    url = 'https://drive.google.com/uc?id=1PpaFM7tjZQ9LuICfNmITrbbJnq4SFGnK'
    output = 'model_vgg_fine_ind_grad.h5'
    gdown.download(url, output, quiet=True)
download_model()

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model_vgg_fine_ind_grad.h5')
    return model

# model = tf.keras.models.load_model('model_resnet_fine_ind.h5')
mapbox_token = 'pk.eyJ1IjoiYWRpdGktMTgiLCJhIjoiY2xsZ2dlcm9zMHRiMzNkcWF2MmFjZTc3biJ9.axO4l5PRwHHn2H3wEE-cEg'

def get_static_map_image(latitude, longitude, api):
 # Replace with your Google Maps API Key
    base_url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
        'center': f'{latitude},{longitude}',
        'zoom': 17,  # You can adjust the zoom level as per your requirement
        'size': '256x276',  # You can adjust the size of the image as per your requirement
        'maptype': 'satellite',
        'key': api,
    }
    response = requests.get(base_url, params=params)
    return response.content

def create_map(zoom_level):
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=zoom_level,
        control_scale=True
    )

    # Add Mapbox tiles with 'Mapbox Satellite' style
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}",
        attr="Mapbox Satellite",
        name="Mapbox Satellite"
    ).add_to(india_map)

    plugins.MousePosition().add_to(india_map)

    return india_map

def add_locations(lat,lon,india_map):
    india_map.location = [lat, lon]

    # Add marker for selected latitude and longitude
    folium.Marker(
        location=[lat, lon],
        popup=f"Latitude: {lat}, Longitude: {lon}",
        icon=folium.Icon(color='blue')
    ).add_to(india_map)



def imgs_input_fn(images):
    img_size = (224, 224)
    img_size_tensor = tf.constant(img_size, dtype=tf.int32)
    images = tf.convert_to_tensor(value = images)
    images = tf.image.resize(images, size=img_size_tensor)
    return images

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_array, heatmap, alpha=0.4):
    img = img_array
    heatmap = np.uint8(255 * heatmap)
    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    return superimposed_img

def main():

    hide_st_style = """
            <style>
            body {
            background-color: black;
            color: white;
        }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    model = load_model()

    st.title("Brick Kiln Detector")
    st.write("This app uses a deep learning model to detect brick kilns in satellite images. The app allows you to select certain area on a map and download the images of brick kilns and non-brick kilns in that region.")

    st.header("Instructions")
    st.write("1. Enter the latitude and longitude of the bounding box in the sidebar.\n"
                 "2. Click on submit and wait for the results to load.\n"
                 "3. Download the images and CSV file using the download buttons below.")


    if 'india_map' not in st.session_state:
        st.session_state.india_map = create_map(5)


    # Initialize variables to store user-drawn polygons
    drawn_polygons = []

    # Specify the latitude and longitude for the rectangular bounding box
    st.sidebar.title("Bounding Box")
    
    box_lat1 = st.sidebar.number_input("Latitude of Bottom-left corner:", value=26.40, step=0.01,on_change=callback_map)
    box_lon1 = st.sidebar.number_input("Longitude of Bottom-Left corner:", value=79.55, step=0.01,on_change=callback_map)
    box_lat2 = st.sidebar.number_input("Latitude of Top-Right corner:", value=26.45, step=0.01,on_change=callback_map)
    box_lon2 = st.sidebar.number_input("Longitude of Top-Right corner:", value=79.60, step=0.01,on_change=callback_map)
    area = np.abs(box_lat2-box_lat1)*np.abs(box_lon2-box_lon1)
    area = round(area,5)
    st.sidebar.write(f"Area of the bounding box is {area} sq units.")

    if area<=0.25:

        # Add the rectangular bounding box to the map
        bounding_box_polygon = folium.Rectangle(
            bounds=[[box_lat1, box_lon1], [box_lat2, box_lon2]],
            color='red',
            fill=True,
            fill_opacity=0.2,
        )
        bounding_box_polygon.add_to(st.session_state.india_map)
        drawn_polygons.append(bounding_box_polygon.get_bounds())

        df = pd.DataFrame(columns = ['Sr.No','Latitude', 'Longitude','P(Brick kiln)'])

        
        # Display the map as an image using st.image()
        folium_static(st.session_state.india_map)
        
        ab = st.secrets["Api_key"]
        # ab = "AIzaSyCBGIlzrt1yWOzXU7L3_2eaSJcxFHiedz0"
        


        if ab and (st.button("Submit",on_click=callback) or st.session_state.button1):
            @st.cache_resource(show_spinner = False)
            def done_before(df,drawn_polygons):
                st.session_state.ab = ab
                image_array_list = []
                latitudes = []
                longitudes = []
                idx = 0
                lat_1 = drawn_polygons[0][0][0]
                lon_1 = drawn_polygons[0][0][1]
                lat_2 = drawn_polygons[0][1][0]
                lon_2 = drawn_polygons[0][1][1]
                delta_lat = 0.0045
                delta_lon = 0.0051
                latitude = lat_1
                longitude = lon_1
                nlat=0
                nlong=0
                while latitude<=lat_2:
                    nlat+=1
                    latitude+=delta_lat

                while longitude<=lon_2:
                    nlong+=1
                    longitude+=delta_lon
                latitude=lat_1
                longitude=lon_1

                progress_text = 'Please wait while we process your request...'
                my_bar = st.progress(0, text=progress_text)


                
                while latitude<=lat_2:
                    while longitude<=lon_2:
                        image_data = get_static_map_image(latitude, longitude, ab)
                        image = Image.open(io.BytesIO(image_data))

                
                        # Get the size of the image (width, height)
                        width, height = image.size
                    
                

                        new_height = height - 20
                
                        # Define the cropping box (left, upper, right, lower)
                        crop_box = (0, 0, width, new_height)
                            
                        # Crop the image
                        image = image.crop(crop_box)

                        new_width = 224
                        new_height = 224

                        # Define the resizing box (left, upper, right, lower)
                        resize_box = (0, 0, new_width, new_height)

                        # Resize the image
                        image = image.resize((new_width, new_height), Image.LANCZOS)

                        if image.mode != 'RGB':
                            image = image.convert('RGB')


                        image_np_array = np.array(image)
                            
                        # image_np_array = np.array(image)
                            
                            
                        image_array_list.append(image_np_array)
                        latitudes.append(latitude)
                        longitudes.append(longitude)
                
                            
                            
                        longitude += delta_lon
                        my_bar.progress(idx , text=progress_text)
                        idx+=(3/(4*nlat*nlong))
                
                        
                            
                        
                    latitude += delta_lat
                    longitude = lon_1
                
                    

                images = np.stack(image_array_list, axis=0)
                    
            
                # images = imgs_input_fn(image_array_list)
                predictions_prob = model.predict(images)
                predictions = [[1 if element >= 0.5 else 0 for element in sublist] for sublist in predictions_prob]
            
                prob_flat_list = [element for sublist in predictions_prob for element in sublist]
                flat_modified_list = [element for sublist in predictions for element in sublist]
                
                indices_of_ones = [index for index, element in enumerate(flat_modified_list) if element == 1]
                indices_of_zeros = [index for index, element in enumerate(flat_modified_list) if element == 0]

            
                

                return indices_of_ones,latitudes,longitudes,image_array_list,indices_of_zeros,images,predictions_prob,flat_modified_list,my_bar,prob_flat_list
            indices_of_ones,latitudes,longitudes,image_array_list,indices_of_zeros,images,predictions_prob,flat_modified_list,my_bar,prob_flat_list=done_before(df,drawn_polygons) 
            temp_dir1 = tempfile.mkdtemp()  # Create a temporary directory to store the images
            with zipfile.ZipFile('images_kiln.zip', 'w') as zipf:
                s_no =1
                for i in indices_of_ones:
                    truncated_float = int(prob_flat_list[i] * 100) / 100
                    temp_df = pd.DataFrame({'Sr.No':[s_no],'Latitude': [latitudes[i]], 'Longitude': [longitudes[i]],'P(Brick kiln)':[truncated_float]})
                    s_no+=1
                    # Concatenate the temporary DataFrame with the main DataFrame
                    df = pd.concat([df, temp_df], ignore_index=True)
                
                    image_filename = f'kiln_{latitudes[i]}_{longitudes[i]}.png'
                    image_path = os.path.join(temp_dir1, image_filename)

                    pil_image = Image.fromarray(image_array_list[i])

                    pil_image.save(image_path, format='PNG')
                    zipf.write(image_path, arcname=image_filename)
                        
                
            
            temp_dir2 = tempfile.mkdtemp()  # Create a temporary directory to store the images
                
            with zipfile.ZipFile('images_no_kiln.zip', 'w') as zipf:
                for i in indices_of_zeros:
                    image_filename = f'kiln_{latitudes[i]}_{longitudes[i]}.png'
                    image_path = os.path.join(temp_dir2, image_filename)

                    pil_image = Image.fromarray(image_array_list[i])

                    pil_image.save(image_path, format='PNG')
                    zipf.write(image_path, arcname=image_filename)
                        
        
            csv = df.to_csv(index=False).encode('utf-8')
                
                    

            count_ones = sum(1 for element in flat_modified_list if element == 1)
            count_zeros = sum(1 for element in flat_modified_list if element == 0)
            my_bar.progress(0.99 , text='Please wait while we process your request...')
            time.sleep(1)
            my_bar.empty()        

            st.write("The number of brick kilns in the selected region is: ", count_ones)
            st.write("The number of non-brick kilns in the selected region is: ", count_zeros)
          

            if count_ones!=0:
                if st.session_state.zoomed_in:
                    indices_of_ones = np.array(indices_of_ones)
                    latitudes = np.array(latitudes)
                    longitudes = np.array(longitudes)
                    lat_brick_kilns = latitudes[indices_of_ones]
                    lon_brick_kilns = longitudes[indices_of_ones]
                    indices_of_ones = indices_of_ones.tolist()
                    latitudes = latitudes.tolist()
                    longitudes = longitudes.tolist()
                    st.session_state.india_map=create_map(13)
                    bounding_box_polygon.add_to(st.session_state.india_map)
                    for Idx in range(len(lat_brick_kilns)):
                        lat = lat_brick_kilns[Idx]
                        lon = lon_brick_kilns[Idx]
                        add_locations(lat,lon,st.session_state.india_map)
                    st.session_state.zoomed_in = False
                    st.experimental_rerun()
            
                # folium_static(india_map)
                st.markdown("### Download options")
                with open('images_kiln.zip', 'rb') as zip_file:
                    zip_data = zip_file.read()
                st.download_button(
                    label="Download Kiln Images",
                    data=zip_data,
                    file_name='images_kiln.zip',
                    mime="application/zip"
                )
                with open('images_no_kiln.zip', 'rb') as zip_file:
                    zip_data = zip_file.read()
                st.download_button(
                    label="Download Non-Kiln Images",
                    data=zip_data,
                    file_name='images_no_kiln.zip',
                    mime="application/zip"
                )
                st.download_button(label =
                    "Download CSV of latitude and longitude of brick kilns",
                    data = csv,
                    file_name = "lat_long.csv",
                    mime = "text/csv"
                    ) 

                # Cleanup: Remove the temporary directory and zip file
                shutil.rmtree(temp_dir1)
                os.remove('images_kiln.zip')
                shutil.rmtree(temp_dir2)
                os.remove('images_no_kiln.zip')
                
                
                ############## GradCAM ##############
                last_conv_layer_name = "block5_conv3"
                st.write("Let's see how well our model is identifying the pattern of brick kilns in the images.")
                for idx in indices_of_ones:

                    st.write("Predicted Probability: ", round(predictions_prob[idx][0],2))

                    # Load and preprocess the original image
                    img_array = images[idx:idx+1]

                    # Create a figure and axes for the images
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.2, 1.44]})

                    # Display the original image
                    axs[0].imshow(images[idx])
                    axs[0].set_title('Original Image',size="xx-large")

                    # Preprocess the image for GradCAM
                    img_array = imgs_input_fn(img_array)
                    
                    # Generate class activation heatmap
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

                    # Generate and display the GradCAM superimposed image
                    grad_fig = save_and_display_gradcam(images[idx], heatmap)
                    grad_plot = axs[1].imshow(grad_fig, cmap='jet', vmin=0, vmax=1)
                    axs[1].set_title('GradCAM Superimposed',size="xx-large")
                    cbar = plt.colorbar(grad_plot, ax=axs[1], pad=0.02, shrink=0.91)  
                    cbar.set_label('Heatmap Intensity')
                    cbar.ax.tick_params(labelsize=30)
                    
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.markdown("### Download options")
                with open('images_no_kiln.zip', 'rb') as zip_file:
                    zip_data = zip_file.read()
                st.download_button(
                    label="Download Non-Kiln Images",
                    data=zip_data,
                    file_name='images_no_kiln.zip',
                    mime="application/zip"
                )
                shutil.rmtree(temp_dir2)
                os.remove('images_no_kiln.zip')

    else:
        st.write(":red[The bounding box area is too big. The area should be less than or equal to 0.25 sq units]")
        st.sidebar.write(":red[The bounding box area is too big. The area should be less than or equal to 0.25 sq units]")


    

if __name__ == "__main__":
    main()
