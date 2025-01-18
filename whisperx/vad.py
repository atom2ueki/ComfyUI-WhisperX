import os
from typing import Callable, Optional, Text, Union
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from pyannote.audio import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, Segment, SlidingWindowFeature

@dataclass
class SegmentX:
    start: float
    end: float
    speaker: Optional[str] = None

    def duration(self) -> float:
        return self.end - self.start

def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None):
    """Load the Voice Activity Detection model from the segmentation-3.0 model file.
    
    Parameters:
        device: Device for model computation ('cuda' or 'cpu')
        vad_onset: Onset threshold for voice activity detection
        vad_offset: Offset threshold for voice activity detection
        use_auth_token: Hugging Face authentication token
    """
    model_dir = os.path.join(os.path.dirname(__file__), "models", "vad")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "segmentation_3_0.bin")
    
    if not os.path.exists(model_path):
        if not use_auth_token:
            raise ValueError(
                "Please provide a valid Hugging Face token. "
                "Visit https://hf.co/settings/tokens to create one and accept "
                "the terms at https://hf.co/pyannote/segmentation-3.0"
            )
        
        print(f"Downloading segmentation-3.0 model to {model_path}")
        model_file = hf_hub_download(
            repo_id="pyannote/segmentation-3.0",
            filename="pytorch_model.bin",
            token=use_auth_token,
            cache_dir=model_dir
        )
        if not os.path.exists(model_path):
            os.symlink(model_file, model_path)
    
    model_config = {
        "segmentation": {
            "architecture": "PyanNet",
            "task": {
                "class": "OverlappedSpeechDetection",
                "params": {
                    "duration": 2.0,
                    "batch_size": 32,
                    "num_workers": 4
                }
            }
        }
    }
    
    vad_model = Model.from_pretrained(
        model_path,
        map_location=device,
        strict=False
    )
    vad_model.task = model_config["segmentation"]["task"]
    
    hyperparameters = {
        "onset": vad_onset,
        "offset": vad_offset,
        "min_duration_on": 0.1,
        "min_duration_off": 0.1
    }
    
    vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
    vad_pipeline.instantiate(hyperparameters)
    
    return vad_pipeline

class Binarize:
    """Binarize detection scores using hysteresis thresholding with min-cut operation."""
    
    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float('inf')
    ):
        self.onset = onset
        self.offset = offset or onset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Process and binarize detection scores."""
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]
            
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start
            
            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        min_score_div_idx = search_after + np.argmin(curr_scores[search_after:])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx+1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx+1:]
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                elif y > self.onset:
                    start = t
                    is_active = True

            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError("This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active

class VoiceActivitySegmentation(VoiceActivityDetection):
    """Pipeline for voice activity segmentation."""
    
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Union[Text, None] = None,
        **inference_kwargs,
    ):
        super().__init__(
            segmentation=segmentation,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs
        )

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection to audio file."""
        hook = self.setup_hook(file, hook=hook)
        
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations = self._segmentation(file)

        return segmentations

def merge_vad(vad_arr, pad_onset=0.0, pad_offset=0.0, min_duration_off=0.0, min_duration_on=0.0):
    """Merge voice activity detection segments with padding and duration constraints."""
    active = Annotation()
    for k, vad_t in enumerate(vad_arr):
        region = Segment(vad_t[0] - pad_onset, vad_t[1] + pad_offset)
        active[region, k] = 1

    if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0:
        active = active.support(collar=min_duration_off)
    
    if min_duration_on > 0:
        for segment, track in list(active.itertracks()):
            if segment.duration < min_duration_on:
                del active[segment, track]
    
    active = active.for_json()
    active_segs = pd.DataFrame([x['segment'] for x in active['content']])
    return active_segs

def merge_chunks(segments, chunk_size, onset: float = 0.5, offset: Optional[float] = None):
    """Merge segments into chunks with specified size constraints."""
    binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
    segments = binarize(segments)
    segments_list = []
    
    for speech_turn in segments.get_timeline():
        segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

    if not segments_list:
        print("No active speech found in audio")
        return []

    curr_start = segments_list[0].start
    curr_end = segments_list[0].end
    merged_segments = []

    for seg in segments_list[1:]:
        if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
            merged_segments.append({
                "start": curr_start,
                "end": curr_end,
                "segments": [(s.start, s.end) for s in segments_list],
            })
            curr_start = seg.start
        curr_end = seg.end

    merged_segments.append({
        "start": curr_start,
        "end": curr_end,
        "segments": [(s.start, s.end) for s in segments_list],
    })
    
    return merged_segments