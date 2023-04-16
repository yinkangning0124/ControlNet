import pytorch_lightning as pl
from torch.utils.data import DataLoader
from  tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Config
resume_path = './models/control_sd15_ini.ckpt'
batchsize = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models, Pytorch Lightning will automatically move them to GPU.
model = create_model(config_path='./models/cldm_v15.yaml')
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Mics
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=4)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

# Train !
trainer.fit(model, dataloader)

# Config
