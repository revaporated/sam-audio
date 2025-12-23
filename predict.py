# Prediction interface for Cog ⚙️
# https://cog.run/python

import sys
sys.path.insert(0, "/src")

import torch
import torchaudio
from cog import BasePredictor, Input, Path
from typing import Optional
import tempfile
import os

from sam_audio import SAMAudio, SAMAudioProcessor


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model was downloaded at build time to this path
        model_path = "/src/weights/sam-audio-large"
        
        self.model = SAMAudio.from_pretrained(model_path).to(self.device).eval()
        self.processor = SAMAudioProcessor.from_pretrained(model_path)

    def predict(
        self,
        audio: Path = Input(description="Audio file to separate sounds from"),
        description: str = Input(
            description="Text description of the sound to isolate (e.g., 'A man speaking', 'A dog barking')"
        ),
        predict_spans: bool = Input(
            description="Automatically predict time spans where the target sound occurs (better for non-ambient sounds)",
            default=False,
        ),
        reranking_candidates: int = Input(
            description="Number of candidates to generate and rerank (higher = better quality, slower)",
            default=1,
            ge=1,
            le=8,
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
            result = self.model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )
        
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
