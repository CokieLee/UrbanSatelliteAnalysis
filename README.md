# Data Science!

## Datasets:

Eurosat

[DOI](10.1109/IGARSS.2018.8519248)

[dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)

download EuroSAT_MS.zip (the full spectral dataset) and extract it into this directory


## Trained Models

### Deepest_CNN



To load the trained model, you need the following lines:
```
from image_models import Deepest_CNN
model = Deepest_CNN(3,10).to(device)
model.load_state_dict(torch.load("Trained_40epoch_Deepest.pth", weights_only=True))
```