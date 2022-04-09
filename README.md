# MLNS-project-datascienthyris
Classification of seizure types using EEG signals, analyzing the measurement with a network approach


To download the data, you need to request credentials:
Register here to get the credentials https://isip.piconepress.com/projects/tuh_eeg/html/request_access.php

Then you can just run the command:

```sh
rsync -auxvL nedc@www.isip.piconepress.com:data/eeg/tuh_eeg_seizure/v1.5.2/ data/
```

# Data


# Models

The architecure of each models are stored in the model.architecture file. For now it includes :
- A classical convolutional architecture
- An attention based architecture

To lauch a training, you can either use the file model.train.py as an executable file, or use the functions implemented in this file in a notebook (an example is given in the train.ipynb notebook). If you run it as an executable, the hyperparameters are taken from the config.yaml file such as :
-  BATCH_SIZE: batch size
-  MODEL_TYPE: type of model you want to train
-  LR: learning rate
-  EPOCHS: length of the training
-  MODEL_ARGS: additional parameters (accordingly to the parameters from the architecture you chose)

In order to visualise the results, you can just execute visualisation.py
