# import necessary libraries
import dash
from dash import html
from dash import dcc
import base64
from io import BytesIO
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import rescale, resize
from skimage.util import crop
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import cv2


##############################################################################
###                INITIALIZE GLOBAL VARIABLES
##############################################################################
# load the trained sign language image detection model
model = tf.keras.models.load_model('modelh5.h5')

# store image and prediction history
pred_history = []



##############################################################################
###                PREDICTION + UTILITY FUNCTIONS
##############################################################################

# function to predict the letter based on the sign language image
def predict_image(image):
    # processing the image to make it suitable for input into the model
    image = resize(image, (28, 28), anti_aliasing=False)
    image = image.reshape([28, 28, 1])
    x = np.array(image)
    # io.imsave('image_edited1.jpg', x)
    x = x.reshape(-1, 28, 28, 1)


    # making the prediction
    predict_x = model.predict(x)
    predictions = np.argmax(predict_x, axis=1)

    classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
               'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    return classes[predictions[0]]



##############################################################################
###                     WEBAPP LAYOUT
##############################################################################

# define the Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# define layout of left column
left_panel = html.Div([html.H2("Sign Language Detection", style={'textAlign': 'center'}),
                       html.Hr(),
                       html.P("Press the button below to click a live picture or use the space below to upload an image.",
                              className="lead", style={'textAlign': 'center',
                                                       'color': 'black'}),
                       html.Button('Click',
                                   id='camera-click-button',
                                   style={'marginBottom': '10px',
                                          'display': 'block',
                                          'marginLeft': 'auto',
                                          'marginRight': 'auto'}),
                       dcc.Upload(id='upload-image-box',
                                  children=html.Div([html.P('Drag and drop or click to select an image to upload.',
                                                            style={'margin': 'auto',
                                                                   'color': 'black'})],
                                                    style={'margin': 'auto'}),
                                  style={
                                      'width': '90%',
                                      'height': '300px',
                                      'borderWidth': '1px',
                                      'borderStyle': 'dashed',
                                      'borderRadius': '5px',
                                      'textAlign': 'center',
                                      'margin': 'auto',
                                      'display': 'flex',
                                      'alignItems': 'center'
                                  },
                                  multiple=True),
                       html.Div(id='lhp-output-box', style={'textAlign': 'center',
                                                            'paddingTop': '40px'})
                       ],
                      style={'padding': '2rem 2rem 2rem 2rem',
                             "backgroundColor": "#E1D6D6"})

# define layout of right column
right_panel = html.Div([html.H2("Output", style={'textAlign': 'center'}),
                        html.Hr(),
                        html.Button('Delete Last Image',
                                    id='delete-button',
                                    style={'marginBottom': '10px',
                                           'display': 'block',
                                           'marginLeft': 'auto',
                                           'marginRight': 'auto'}),
                        html.Button('Reset All Images',
                                    id='reset-button',
                                    style={'marginBottom': '10px',
                                           'display': 'block',
                                           'marginLeft': 'auto',
                                           'marginRight': 'auto'}),
                        html.Div(id='rhp-output-box', style={'textAlign': 'center',
                                                             'paddingTop': '40px'})
                        ],
                       style={'padding': '2rem 2rem 2rem 2rem'})

# define the layout of the app
app.layout = html.Div(children=dbc.Row([dbc.Col(left_panel, width=4),
                                        dbc.Col(right_panel)]))



##############################################################################
###                   INTERACTION FUNCTIONS
##############################################################################

# callback function to display the output when an image is uploaded
@app.callback(
    Output('lhp-output-box', 'children', allow_duplicate=True),
    Output('rhp-output-box', 'children', allow_duplicate=True),
    Input('upload-image-box', 'contents'),
    prevent_initial_call=True
)
def update_upload_image(contents):
    if contents is None:
        return dash.no_update

    if contents is not None:
        prediction = []
        for content in contents:
            # decode the uploaded image
            _, content_string = content.split(',')
            image = io.imread(BytesIO(base64.b64decode(content_string)),
                              as_gray=True)

            # predicting letter and storing it
            current_prediction = html.Div([html.Img(src=content,
                                                    # Set fixed width and height
                                                    style={'width': '100px',
                                                           'height': '100px',
                                                           'objectFit': 'contain'}),
                                           html.H2(predict_image(image))
                                           ])

            prediction.append(current_prediction)
            pred_history.append(current_prediction)

        # creating left panel output
        lhp_output = html.Div([html.P("Uploaded Image(s) and Predicted Letter(s):",
                                      style={'color': 'black',
                                             'textAlign': 'center'}),
                               html.Div(prediction,
                                        style={'display': 'flex',
                                               'flexWrap': 'wrap',
                                               'alignItems': 'center',
                                               'justifyContent': 'center'})])

        # creating right panel output
        rhp_output = html.Div(pred_history,
                              style={'display': 'flex',
                                     'flexWrap': 'wrap',
                                     'justifyContent': 'flex-start',
                                     'gap': '10px'})


        # display the uploaded image and its predicted letter
        return lhp_output, rhp_output


# callback function to display the output when an image is clicked
@app.callback(
    Output('lhp-output-box', 'children'),
    Output('rhp-output-box', 'children'),
    Input('camera-click-button', 'n_clicks')
)
def update_camera_image(n_clicks):
    if n_clicks is None:
        return dash.no_update

    # capturing an image from the webcam
    cap = cv2.VideoCapture(1) #try 0 if 1 does not work
    while True:
        cv2.waitKey(500)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('image.jpg', frame)
            break
    cap.release()

    # reading the saved image and cropping it for the model
    image = io.imread('image.jpg', as_gray=True)
    crop_size = int((image.shape[1]-image.shape[0])/2)
    image = crop(image, ((0, 0), (crop_size, crop_size)), copy=False)

    # adjusting, saving, and opening the clicked image to show to the user
    io.imsave('image_edited.jpg', resize(image, (480, 480), anti_aliasing=False))
    with open("image_edited.jpg", "rb") as img_file:
        content = 'data:image/jpeg;base64,' + base64.b64encode(img_file.read()).decode('utf-8')

    # predicting letter and storing it
    current_prediction = html.Div([html.Img(src=content,
                                            # Set fixed width and height
                                            style={'width': '100px',
                                                   'height': '100px',
                                                   'objectFit': 'contain'}),
                                   html.H2(predict_image(image))
                                   ])

    pred_history.append(current_prediction)

    # creating left panel output
    lhp_output = html.Div([html.P("Uploaded Image and Predicted Letter:",
                                  style={'color': 'black',
                                         'textAlign': 'center'}),
                           html.Div([current_prediction],
                                    style={'display': 'flex',
                                           'flexWrap': 'wrap',
                                           'alignItems': 'center',
                                           'justifyContent': 'center'})])

    # creating right panel output
    rhp_output = html.Div(pred_history,
                          style={'display': 'flex',
                                 'flexWrap': 'wrap',
                                 'justifyContent': 'flex-start',
                                 'gap': '10px'})

    # display the uploaded image and its predicted letter
    return lhp_output, rhp_output


# callback function to reset all prediction history
@app.callback(
    Output('lhp-output-box', 'children', allow_duplicate=True),
    Output('rhp-output-box', 'children', allow_duplicate=True),
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_all(n_clicks):
    if n_clicks is None:
        return dash.no_update

    pred_history.clear()

    return html.Div(style={'display': 'flex',
                           'flexWrap': 'wrap',
                           'alignItems': 'center',
                           'justifyContent': 'center'}), \
           html.Div(pred_history,
                    style={'display': 'flex',
                           'flexWrap': 'wrap',
                           'justifyContent': 'flex-start',
                           'gap': '10px'})


# callback function to delete last input
@app.callback(
    Output('lhp-output-box', 'children', allow_duplicate=True),
    Output('rhp-output-box', 'children', allow_duplicate=True),
    Input('delete-button', 'n_clicks'),
    prevent_initial_call=True
)
def delete_latest(n_clicks):
    if n_clicks is None:
        return dash.no_update

    try:
        pred_history.pop()
    except:
        pass

    return html.Div(style={'display': 'flex',
                           'flexWrap': 'wrap',
                           'alignItems': 'center',
                           'justifyContent': 'center'}),\
           html.Div(pred_history,
                    style={'display': 'flex',
                           'flexWrap': 'wrap',
                           'justifyContent': 'flex-start',
                           'gap': '10px'})



##############################################################################
###              MAIN FUNCTION TO RUN SCRIPT
##############################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
