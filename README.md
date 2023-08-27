# Federated-Learning

This is a demo project for federated learning using **Socket Programming**. I have used [PyGAD](https://pygad.readthedocs.io/en/latest/pygad.html) module for making the neural network model using Genetic Algorithm. The neural network model is generated to learn the OR problem with inputs and outputs as numpy arrays. The model data and parameters are sent from server to client using a socket and then the client trains the model on its own data which is specified in the `client.py` file and sends the updated model parameters using the same connection. 

# Files

1. `server.py` -> This file contains the code for the server which initiates the connection and sends the initial model parameters. I have also included *threading*, which allows server to accept multiple connections att the same time.
2. `client.py` -> This file contains code for client which accepts initial model parameters from the server via a socket connection and then trains the model onnn its own data. The trained model is then sent to the server.

# Intalling all the dependencies and running the project

Run the command `pip3 install -r requirements.txt` on linux and `pip install -r requirements.txt` on windows to install all the dependencies.
To start the server run `python3 server.py` on linux and `python server.py` on windows. After starting the server run the client using the `python3 client.py`.

**NOTE**: For testing multiple active connections, one can create another `client11.py` file which contains the same code as `client.py` and then running the files simultaneously to create multiple connections.
