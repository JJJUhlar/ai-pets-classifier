# AI Pet Classifier
AI Pet classifier can distinguish between more than 30 different types of pet. Users can run AI Pet Classifier locally and recieve predicitions distinguishing between over 30 different pets!

AI Pet Classifier is based on a ResNet18 model and trained further on the Oxford Pets III dataset. Libraries used during training include PyTorch, FastAI, Pandas, Numpy. Meanwhile, the server is implemented using Flask.

The app is ready to use locally and also contains config files to be run via heroku if desired.

## Setup and Use locally

1. Clone and fork the repo.
2. Open the repo in your terminal
3. Activate python environment by using `$ source venv/bin/activate` or for windows `source venv\Scripts\activate`.
4. Run the app by running `python3 app.py`
5. You should see something like the following: `running on 127.0.0.1:5000` Open this address in your browser.
6. You can now upload pictures of pets and recieve classifications using the buttons!!!
7. Uploaded images will be stored in the `./images/` directory.