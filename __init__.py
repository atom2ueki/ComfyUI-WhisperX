import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
WEB_DIRECTORY = "./web"
from .nodes import WhisperX, PreViewSRT, SRTToString

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "WhisperX": WhisperX,
    "PreViewSRT": PreViewSRT,
    "SRTToString": SRTToString
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperX": "WhisperX Node",
    "PreViewSRT": "PreView SRT",
    "SRTToString": "SRT to String"
}