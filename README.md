# Landmark Recognition
An end-to-end application to predict landmarks from images, retrieve geolocation details, and visualize results on an interactive map.

## Features
- Simple repsonsive UI.
- It will give you the full address of Landmark
- It will provide you the `Latitude` & `Longitude` of predicted landmark.
- It will plot the predicted landmark on the Map.

## Model & Data Source
- Leveraged a Pretrained TensorFlow-Hub Model ([`landmarks_classifier_asia_V1/1`](https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1)) for Landmark Classification
- Supports roughly 98,961 landmark classes covering Asia’s most iconic sites.
- This model was trained on [`Google Landmarks Dataset V2`](https://ai.googleblog.com/2019/05/announcing-google-landmarks-v2-improved.html).
- 
## Usage

- Clone this repository.
- Open CMD in working directory.
- Run following command.

  ```
  pip install -r requirements.txt
  ```
- `LM_Detection.py` is the main Python file of Streamlit Web-Application. 
- To run app, write following command in CMD. or use any IDE.

  ```
  streamlit run lm_recog.py
  ```

- For more explanation of this project see the tutorial on Machine Learning Hub YouTube channel.

## Screenshots

<img src="https://github.com/Spidy20/LandMark_Detection/blob/master/s1.PNG">
<img src="https://github.com/Spidy20/LandMark_Detection/blob/master/s2.PNG">
