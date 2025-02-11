from datasets.data_loader import NumerosityDataModule
import pytorch_lightning as pl
from models.multimodal_idbn import iMDBN,iDBN
import torch

data_module = NumerosityDataModule('NumStim_1to20_100x100_TR.mat', batch_size=128)
data_module.setup()
params = {
  "LEARNING_RATE": 0.15,
     "WEIGHT_PENALTY": 0.0001,
     "INIT_MOMENTUM": 0.7,
     "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
     "LEARNING_RATE_DYNAMIC": True,
}
print("sdone")
model = iDBN(params,layer_sizes=[500,500])
trainer = pl.Trainer(max_epochs=10, devices=1 if torch.cuda.is_available() else 0)
trainer.fit(model, data_module)

