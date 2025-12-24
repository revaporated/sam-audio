# Prediction interface for Cog ⚙️
# https://cog.run/python

import sys
sys.path.insert(0, "/src")

import torch
import torchaudio
from cog import BasePredictor, Input, Path
import tempfile
import os

from sam_audio import SAMAudio, SAMAudioProcessor


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model was downloaded at build time to this path
        model_path = "/weights/sam-audio-base"
        
        # Disable rankers and span predictor to speed up loading
        # (they require additional large models like ImageBind and CLAP)
        self.model = SAMAudio.from_pretrained(
            model_path,
            visual_ranker=None,
            text_ranker=None,
            span_predictor=None,
        ).to(self.device).eval()
        self.processor = SAMAudioProcessor.from_pretrained(model_path)

    def predict(
        self,
        audio: Path = Input(description="Audio file to separate sounds from"),
        description: str = Input(
            description="Text description of the sound to isolate (e.g., 'A man speaking', 'A dog barking')"
        ),
    ) -> list[Path]:
        """Run audio source separation based on text description"""
        
        # Process input
        batch = self.processor(
            audios=[str(audio)],
            descriptions=[description],
        ).to(self.device)
        
        # Run separation
        with torch.inference_mode():
            result = self.model.separate(batch)
        
        # Save outputs
        output_dir = tempfile.mkdtemp()
        sample_rate = self.processor.audio_sampling_rate
        
        target_path = Path(os.path.join(output_dir, "target.wav"))
        residual_path = Path(os.path.join(output_dir, "residual.wav"))
        
        torchaudio.save(
            str(target_path),
            result.target[0].unsqueeze(0).cpu(),
            sample_rate,
        )
        torchaudio.save(
            str(residual_path),
            result.residual[0].unsqueeze(0).cpu(),
            sample_rate,
        )
        
        return [target_path, residual_path]
