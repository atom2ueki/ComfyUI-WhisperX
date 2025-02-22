import os
import srt
import torch
import time
import whisperx
import folder_paths
import cuda_malloc
import translators as ts
from tqdm import tqdm
from datetime import timedelta
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class PreViewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }

    CATEGORY = "ATOM2UEKI_WhisperX"

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r') as f:
            srt_content = f.read()
        return {"ui": {"srt":[srt_content,srt_name,dir_name]}}


class SRTToString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"

    CATEGORY = "ATOM2UEKI_FishSpeech"

    def read(self,srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r', encoding="utf-8") as f:
            srt_content = f.read()
        return (srt_content,)
    
class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3","distil-large-v3","large-v2", "large-v3-turbo"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing', 'caiyun', 'cloudTranslation',
                           'deepl', 'elia', 'google', 'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
                           'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai', 'modernMt', 'myMemory',
                           'niutrans', 'papago', 'qqFanyi', 'qqTranSmart', 'reverso', 'sogou', 'sysTran',
                           'tilde', 'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex', 'yeekit',
                           'youdao']
        lang_list = ["auto", "zh", "en", "ja", "ko", "ru", "fr", "de", "es", "pt", "it", "ar"]
        return {"required": {
            "audio": ("AUDIOPATH",),
            "model_type": (model_list, {"default": "large-v3"}),
            "batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            "if_multiple_speaker": ("BOOLEAN", {"default": False}),
            "use_auth_token": ("STRING", {"default": "", "multiline": False, "placeholder": "Enter your Hugging Face token"}),
            "source_language": (lang_list, {"default": "auto"}),
            "if_translate": ("BOOLEAN", {"default": False}),
            "translator": (translator_list, {"default": "alibaba"}),
            "to_language": (lang_list[1:], {"default": "en"})
        }}

    CATEGORY = "ATOM2UEKI_WhisperX"
    RETURN_TYPES = ("SRT", "SRT")
    RETURN_NAMES = ("ori_SRT", "trans_SRT")
    FUNCTION = "get_srt"

    def validate_auth_token(self, use_auth_token):
        """Validate and prepare the authentication token."""
        if not use_auth_token or use_auth_token.strip() == "":
            raise ValueError("Please provide a valid Hugging Face authentication token. Visit https://hf.co/settings/tokens to create one.")
        return use_auth_token.strip()

    def get_srt(self, audio, model_type, batch_size, if_multiple_speaker,
                use_auth_token, source_language, if_translate, translator, to_language):
        compute_type = "float16"
        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"

        if model_type == "large-v3-turbo":
            model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"

        # Validate token if speaker diarization is enabled
        if if_multiple_speaker:
            use_auth_token = self.validate_auth_token(use_auth_token)

        try:
            # 1. Load and transcribe with whisper
            vad_options = {"use_auth_token": use_auth_token} if use_auth_token else {}
            model = whisperx.load_model(
                model_type,
                device,
                compute_type=compute_type,
                language=None if source_language == "auto" else source_language,
                vad_options=vad_options
            )

            print(f"Loading audio file: {audio}")
            audio_data = whisperx.load_audio(audio)
            result = model.transcribe(audio_data, batch_size=batch_size)

            detected_language = result["language"]
            print(f"Detected language: {detected_language}")

            # 2. Align whisper output
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=device
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio_data,
                    device,
                    return_char_alignments=False
                )
                
                del model_a
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as align_error:
                print(f"Warning: Alignment failed - {str(align_error)}")
                print("Continuing with unaligned transcription...")

            # 3. Speaker diarization
            if if_multiple_speaker:
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=use_auth_token,
                        device=device
                    )
                    diarize_segments = diarize_model(audio_data)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    del diarize_model
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as diarize_error:
                    print(f"Speaker diarization failed: {str(diarize_error)}")
                    print("Continuing without speaker labels...")

            # 4. Generate output paths
            timestamp = time.time()
            srt_path = os.path.join(out_path, f"{timestamp}_{base_name}.srt")
            trans_srt_path = os.path.join(out_path, f"{timestamp}_{base_name}_{to_language}.srt")

            # 5. Process segments
            srt_line = []
            trans_srt_line = []
            
            print("Processing segments...")
            for i, res in enumerate(tqdm(result["segments"], desc="Creating SRT")):
                start = timedelta(seconds=res['start'])
                end = timedelta(seconds=res['end'])
                speaker_name = res.get("speaker", "")[-1] if "speaker" in res else "0"
                content = res['text']
                
                srt_line.append(srt.Subtitle(
                    index=i+1,
                    start=start,
                    end=end,
                    content=f"Speaker {speaker_name}: {content}" if speaker_name != "0" else content
                ))

                # 6. Handle translation if requested
                if if_translate:
                    try:
                        translated_content = ts.translate_text(
                            query_text=content,
                            translator=translator,
                            to_language=to_language
                        )
                        trans_srt_line.append(srt.Subtitle(
                            index=i+1,
                            start=start,
                            end=end,
                            content=f"Speaker {speaker_name}: {translated_content}" if speaker_name != "0" else translated_content
                        ))
                    except Exception as translate_error:
                        print(f"Translation failed for segment {i+1}: {str(translate_error)}")
                        trans_srt_line.append(srt_line[-1])

            # 7. Write output files
            print(f"Writing SRT files to {out_path}")
            with open(srt_path, 'w', encoding="utf-8") as f:
                f.write(srt.compose(srt_line))

            if if_translate:
                with open(trans_srt_path, 'w', encoding="utf-8") as f:
                    f.write(srt.compose(trans_srt_line))
            else:
                trans_srt_path = srt_path

            print("Processing completed successfully")
            return (srt_path, trans_srt_path)

        except Exception as e:
            raise RuntimeError(f"Error in WhisperX processing: {str(e)}")