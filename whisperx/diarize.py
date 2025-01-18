import os
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union, Dict, Any
import torch
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from .audio import load_audio, SAMPLE_RATE

@dataclass
class Segment:
    start: float
    end: float
    speaker: Optional[str] = None

    def duration(self) -> float:
        return self.end - self.start

class DiarizationPipeline:
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir: Optional[str] = None
    ):
        if isinstance(device, str):
            device = torch.device(device)
            
        if not use_auth_token:
            raise ValueError(
                "Please provide a valid Hugging Face token. "
                "Visit https://hf.co/settings/tokens to create one and accept "
                "the user conditions at https://hf.co/pyannote/speaker-diarization-3.1"
            )
            
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisperx", "diarization")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            print(f"Loading diarization model {model_name}...")
            # This will download the model if it's not already cached
            self.model = Pipeline.from_pretrained(
                model_name,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir
            ).to(device)
            print("Diarization model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization model: {str(e)}")

    def __call__(
        self, 
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> pd.DataFrame:
        if isinstance(audio, str):
            audio = load_audio(audio)

        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        segments = self.model(
            audio_data, 
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=['segment', 'label', 'speaker']
        )
        
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        return diarize_df

def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Dict[str, Any],
    fill_nearest: bool = False
) -> Dict[str, Any]:
    transcript_segments = transcript_result["segments"]
    
    for seg in transcript_segments:
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - \
                                   np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - \
                             np.minimum(diarize_df['start'], seg['start'])

        dia_tmp = diarize_df if fill_nearest else diarize_df[diarize_df['intersection'] > 0]
        
        if len(dia_tmp) > 0:
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker

        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - \
                                               np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - \
                                        np.minimum(diarize_df['start'], word['start'])

                    dia_tmp = diarize_df if fill_nearest else diarize_df[diarize_df['intersection'] > 0]
                    
                    if len(dia_tmp) > 0:
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    return transcript_result