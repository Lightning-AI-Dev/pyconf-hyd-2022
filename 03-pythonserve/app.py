# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"
# !pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q

import lightning as L
from lightning.app.components.serve import Image, PythonServer
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class SDServe(PythonServer):
    def __init__(self, sd_variant="sd1", **kwargs):
        super().__init__(input_type=Prompt, output_type=Image, **kwargs)
        self.sd_variant = sd_variant

    def setup(self, *args, **kwargs) -> None:
        from stable_diffusion_inference import create_text2image

        self._model = create_text2image(self.sd_variant)

    def predict(self, request: Prompt):
        return {"image": self._model(request.prompt)}


app = L.LightningApp(SDServe())
