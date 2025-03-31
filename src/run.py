from datasets.zipfian_dataset import ZipfianDataModule
import pytorch_lightning as pl
from models.multimodal_idbn import iMDBN,iDBN
import torch

data_module = ZipfianDataModule("/home/student/Desktop/Groundeep/training_tensors", batch_size=10,subsample_fraction=0.2,num_workers=3)



data_module.setup()
params = {
  "LEARNING_RATE": 0.15,
     "WEIGHT_PENALTY": 0.0001,
     "INIT_MOMENTUM": 0.7,
     "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
     "LEARNING_RATE_DYNAMIC": True,
}
print("sdone")

layer_sizes = [500,500]
model = iDBN(params,layer_sizes=[data_module.data_shape]+layer_sizes,batch_size=data_module.batch_size) # it should be initialized as a list 
trainer = pl.Trainer(max_epochs=200, devices=1 if torch.cuda.is_available() else 0,reload_dataloaders_every_n_epochs=0 )
trainer.fit(model, data_module.train_dataloader())

