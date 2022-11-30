# app.py
import lightning as L
from lightning.app.components import LightningTrainerMultiNode
import mnist_lit_model as mnist_utils

class LightningTrainerDistributed(L.LightningWork):
    def run(self):
        model = mnist_utils.LitClassifier()
        datamodule = mnist_utils.MyDataModule()
        trainer = L.Trainer(max_epochs=10, strategy="ddp")
        trainer.fit(model, datamodule=datamodule)

# 8 GPU: (2 nodes of 4 x v100)
component = LightningTrainerMultiNode(
    LightningTrainerDistributed,
    num_nodes=1,
    cloud_compute=L.CloudCompute("gpu"),
)
app = L.LightningApp(component)
