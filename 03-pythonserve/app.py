# !pip install taming-transformers-rom1504 -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"
# !pip install lightning-api-access

import base64
import io

import lightning as L
from lightning.app.components.serve import Image, PythonServer
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class SDServe(PythonServer):
    def __init__(self, sd_variant="sd1", **kwargs):
        super().__init__(input_type=Prompt, output_type=Image, **kwargs)
        self.sd_variant = sd_variant

    def serialize(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def setup(self, *args, **kwargs) -> None:
        from stable_diffusion_inference import create_text2image

        self._model = create_text2image(self.sd_variant)

    def predict(self, request: Prompt):
        prediction = self._model(request.prompt)
        return {"image": self.serialize(prediction)}


app = L.LightningApp(SDServe(cloud_compute=L.CloudCompute("gpu", disk_size=30)))
