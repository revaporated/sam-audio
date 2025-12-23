# Prediction interface for Cog ⚙️
# https://cog.run/python

import sys
sys.path.insert(0, "/src")

print("DEBUG: Starting imports...", flush=True)

import torch
print("DEBUG: torch imported", flush=True)

import torchaudio
print("DEBUG: torchaudio imported", flush=True)

from cog import BasePredictor, Input, Path
print("DEBUG: cog imported", flush=True)

from typing import Optional
import tempfile
import os

print("DEBUG: About to import sam_audio...", flush=True)
from sam_audio import SAMAudio, SAMAudioProcessor
print("DEBUG: sam_audio imported", flush=True)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("DEBUG: setup() started", flush=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: device = {self.device}", flush=True)
        
        # Check if model files exist
        model_path = "/weights/sam-audio-base"
        print(f"DEBUG: Checking model path: {model_path}", flush=True)
        
        if os.path.exists(model_path):
            print(f"DEBUG: Model path exists, contents: {os.listdir(model_path)}", flush=True)
        else:
            print(f"DEBUG: ERROR - Model path does not exist!", flush=True)
        
        # Check if ImageBind weights exist
        imagebind_path = "/src/.checkpoints/imagebind_huge.pth"
        if os.path.exists(imagebind_path):
            print(f"DEBUG: ImageBind weights found at {imagebind_path}", flush=True)
        else:
            print(f"DEBUG: WARNING - ImageBind weights not found at {imagebind_path}", flush=True)
        
        print("DEBUG: Loading model...", flush=True)
        self.model = SAMAudio.from_pretrained(model_path).to(self.device).eval()
        print("DEBUG: Model loaded", flush=True)
        
        self.processor = SAMAudioProcessor.from_pretrained(model_path)
        print("DEBUG: Processor loaded", flush=True)
        print("DEBUG: setup() complete!", flush=True)

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
