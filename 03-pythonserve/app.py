# !pip install git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import lightning as L
from lightning_diffusion import BaseDiffusion
from stable_diffusion_inference import create_text2image


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        self.model = create_text2image()

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
