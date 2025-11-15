# Data Science!

## Datasets

We used two primary datasets for our project. Eurosat for training and 

### Eurosat - [DOI](https://doi.org/10.5281/zenodo.7711810) - [dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)

Download EuroSAT_MS.zip and EuroSAT_RGB.zip and extract them into this directory

### Berlin/Beijing - [DOI](https://doi.org/10.1016/j.rse.2023.113856) - [dataset](https://github.com/danfenghong/RSE_Cross-city)

## Data Models

### Model Definitions
The [image_models.py](image_models.py) file contains definitions for several Convolutional Neural Networks, but the main models of interest are Basic_CNN, Deep_CNN, and Deepest_CNN

For this project the in_channels for any given model should either 3 or 13, depending on if you are using the RGB data or the MS (Multi-Spectral) data.

#### Basic_CNN

- 2 Convolutional layers
- 2 pooling layers
- A fully connected layer
- ReLU applied after each convolutional layer
- Softmax applied on last layer (this was removed in the Deepest_CNN model)

#### Deep_CNN

- 3 Convolutional layers
- 3 pooling layers
- A fully connected layer
- ReLU applied after each convolutional layer
- Softmax applied on last layer (this was removed in the Deepest_CNN model)

#### Deepest_CNN

- 4 Convolutional layers
- 4 pooling layers
- A fully connected layer
- ReLU applied after each convolutional layer
- Batch Normalization applied after each layer



### Saved States
In the [TrainedModels](TrainedModels) directory there are saved state dictionaries for the Basic_CNN, Deep_CNN, and Deepest_CNN after 40 training epochs for both the MS dataset and the RGB dataset.


An example of how to load a trained model:
```python
from image_models import Deepest_CNN
model = Deepest_CNN(3,10).to(device)
model.load_state_dict(torch.load("TrainedModels/RGB_Deepest_40epoch_Trained.pth", weights_only=True))
```
This example loads a trained RGB Deepest_CNN model, and uses an input of 3 channels for RGB data. 

Device needs to be either `"cuda"` or `"cpu"` depending on your hardware 

### Model Training
The primary script for training image models is [image_model_trainer.py](image_model_trainer.py).

#### Configuration Section
##### Model imports
The beginning of main in the training script looks like this:
```python
    # Import your model here after adding it to image_models.py 
    from image_models import Basic_CNN, Deep_CNN, Deepest_CNN
    model = Deepest_CNN(13,10).to(device)
```
This is where CNN models are imported from image_models.py. The first value given to the model should either be 3 or 13 depending on if you are using RGB or MS data.
##### Data loader
```python
    # for data loader:
    dl_batch_size = 32 # sort of hardware specifc
    dl_num_cores = 4 # hardware specific, change this to the number of cores on your cpu
```
These are variables used in the dataloader to set the batch size and number of cores for multiprocessing.

##### Image data normalization
```python
    # Do we want to normalize the dataset based off of the per-pixel average and stdev?
    Do_Image_Normalization = True
```
This is for image normalization, and should remain true unless you want to see what happens when it is turned off. (spoiler: it makes the models perform much worse)
##### Dataset paths and image type
```python
    # image file paths
    image_dir = 'EuroSAT_MS'
    # image type (either set this to 'MS' or 'RGB')
    image_type = 'MS'
```
This is where you tell the code whether to train on MS or RGB data. The image_dir needs to be the directory you unpacked the EuroSAT dataset.
##### Training parameters
```python
    # Training parameters
    num_epochs=40
    learnrate = 0.001
    save_interval = 1
    saved_model_states = "Deepest_CNN"
```
This is where you set the number of training epochs, the learnrate of the training function, how often to save the intermediary model states, as well as the output directory for all the model states to be saved.

#### Training script outputs
Running the script will result in a directory containing the saved state dictionaries of the model you trained at various epochs decided by `save_interval` as well as the initial model state dictionary, and the final model state dictionary. [example_training_directory](example_training_directory) shows this for a Deep_CNN model trained for `num_epochs = 40` and `save_interval = 1`. 

The training script will also output test accuracies in csv format for the model at each of the intermediate states. Examples of this can be found in the [TrainedModelTestAccuracies](TrainedModelTestAccuracies) directory

## Repository Files

### [band_importance.py](band_importance.py)

### [deep_cam.ipynb](deep_cam.py)

### [feature_visualize.ipynb](feature_visualize.ipynb)

### [grad_cam.py](grad_cam.py)

### [PlotTrainTestAccuracy.ipynb](PlotTrainTestAccuracy.ipynb)


