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
        with open(srt, 'r') as f:
            srt_content = f.read()
        return (srt_content,)


class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3", "distil-large-v3", "large-v2"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
            'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
            'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
            'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
            'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
            'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
            'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
            'yeekit', 'youdao']
        lang_list = ["auto", "zh", "en", "ja", "ko", "ru", "fr", "de", "es", "pt", "it", "ar"]
        return {"required": {
            "audio": ("AUDIOPATH",),
            "model_type": (model_list, {
                "default": "large-v3"
            }),
            "batch_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 32
            }),
            "if_multiple_speaker": ("BOOLEAN", {
                "default": False
            }),
            "use_auth_token": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "HF token (only needed for speaker detection)"
            }),
            "source_language": (lang_list, {
                "default": "auto"
            }),
            "if_translate": ("BOOLEAN", {
                "default": False
            }),
            "translator": (translator_list, {
                "default": "alibaba"
            }),
            "to_language": (lang_list[1:], {
                "default": "en"
            })
        }}

    CATEGORY = "ATOM2UEKI_WhisperX"
    RETURN_TYPES = ("SRT", "SRT")
    RETURN_NAMES = ("ori_SRT", "trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(self, audio, model_type, batch_size, if_multiple_speaker,
                use_auth_token, source_language, if_translate, translator, to_language):
        compute_type = "float16"
        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_supported() else "cpu"

        try:
            # 1. Load and transcribe with whisper
            model = whisperx.load_model(
                model_type,
                device,
                compute_type=compute_type,
                language=None if source_language == "auto" else source_language,
                asr_options={"hybrid_forward": True}
            )
            
            audio_data = whisperx.load_audio(audio)
            result = model.transcribe(audio_data, batch_size=batch_size)
            
            # Get detected language if auto was selected
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
            except Exception as e:
                print(f"Warning: Alignment failed - {str(e)}")
                print("Continuing with unaligned transcription...")

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
                except Exception as e:
                    print(f"Speaker diarization failed: {str(e)}")
                    print("Continuing without speaker labels...")

            # Generate output paths
            timestamp = time.time()
            srt_path = os.path.join(out_path, f"{timestamp}_{base_name}.srt")
            trans_srt_path = os.path.join(out_path, f"{timestamp}_{base_name}_{to_language}.srt")

            # Process segments
            srt_line = []
            trans_srt_line = []
            
            for i, res in enumerate(tqdm(result["segments"], desc="Processing segments...")):
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
                    except Exception as e:
                        print(f"Translation failed for segment {i+1}: {str(e)}")
                        trans_srt_line.append(srt_line[-1])

            # Write output files
            with open(srt_path, 'w', encoding="utf-8") as f:
                f.write(srt.compose(srt_line))

            if if_translate:
                with open(trans_srt_path, 'w', encoding="utf-8") as f:
                    f.write(srt.compose(trans_srt_line))
            else:
                trans_srt_path = srt_path

            return (srt_path, trans_srt_path)

        except Exception as e:
            raise RuntimeError(f"Error in WhisperX processing: {str(e)}")
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3", "distil-large-v3", "large-v2"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
            'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
            'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
            'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
            'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
            'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
            'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
            'yeekit', 'youdao']
        lang_list = ["auto", "zh", "en", "ja", "ko", "ru", "fr", "de", "es", "pt", "it", "ar"]
        return {"required": {
            "audio": ("AUDIOPATH",),
            "model_type": (model_list, {
                "default": "large-v3"
            }),
            "batch_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 32
            }),
            "if_multiple_speaker": ("BOOLEAN", {
                "default": False
            }),
            "use_auth_token": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "Enter your Hugging Face token"
            }),
            "source_language": (lang_list, {
                "default": "auto"
            }),
            "if_translate": ("BOOLEAN", {
                "default": False
            }),
            "translator": (translator_list, {
                "default": "alibaba"
            }),
            "to_language": (lang_list[1:], {
                "default": "en"
            })
        }}

    CATEGORY = "ATOM2UEKI_WhisperX"
    RETURN_TYPES = ("SRT", "SRT")
    RETURN_NAMES = ("ori_SRT", "trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(self, audio, model_type, batch_size, if_multiple_speaker,
                use_auth_token, source_language, if_translate, translator, to_language):
        compute_type = "float16"
        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_supported() else "cpu"

        # Validate auth token
        if not use_auth_token or use_auth_token.strip() == "":
            raise ValueError("Please provide a valid Hugging Face authentication token. Visit https://hf.co/settings/tokens to create one.")
        
        try:
            # 1. Load and transcribe with whisper
            model = whisperx.load_model(
                model_type,
                device,
                compute_type=compute_type,
                language=None if source_language == "auto" else source_language,
                asr_options={"hybrid_forward": True},
                vad_options={"use_auth_token": use_auth_token}
            )
            
            audio_data = whisperx.load_audio(audio)
            result = model.transcribe(audio_data, batch_size=batch_size)
            
            # Get detected language if auto was selected
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
            except Exception as e:
                print(f"Warning: Alignment failed - {str(e)}")
                print("Continuing with unaligned transcription...")

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
                except Exception as e:
                    print(f"Speaker diarization failed: {str(e)}")
                    print("Continuing without speaker labels...")

            # Generate output paths
            timestamp = time.time()
            srt_path = os.path.join(out_path, f"{timestamp}_{base_name}.srt")
            trans_srt_path = os.path.join(out_path, f"{timestamp}_{base_name}_{to_language}.srt")

            # Process segments
            srt_line = []
            trans_srt_line = []
            
            for i, res in enumerate(tqdm(result["segments"], desc="Processing segments...")):
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
                    except Exception as e:
                        print(f"Translation failed for segment {i+1}: {str(e)}")
                        trans_srt_line.append(srt_line[-1])

            # Write output files
            with open(srt_path, 'w', encoding="utf-8") as f:
                f.write(srt.compose(srt_line))

            if if_translate:
                with open(trans_srt_path, 'w', encoding="utf-8") as f:
                    f.write(srt.compose(trans_srt_line))
            else:
                trans_srt_path = srt_path

            return (srt_path, trans_srt_path)

        except Exception as e:
            raise RuntimeError(f"Error in WhisperX processing: {str(e)}")
        
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3", "distil-large-v3", "large-v2"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
            'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
            'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
            'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
            'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
            'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
            'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
            'yeekit', 'youdao']
        lang_list = ["auto", "zh", "en", "ja", "ko", "ru", "fr", "de", "es", "pt", "it", "ar"]
        return {"required": {
            "audio": ("AUDIOPATH",),
            "model_type": (model_list, {
                "default": "large-v3"
            }),
            "batch_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 32
            }),
            "if_multiple_speaker": ("BOOLEAN", {
                "default": False
            }),
            "use_auth_token": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "Enter your Hugging Face token"
            }),
            "source_language": (lang_list, {
                "default": "auto"
            }),
            "if_translate": ("BOOLEAN", {
                "default": False
            }),
            "translator": (translator_list, {
                "default": "alibaba"
            }),
            "to_language": (lang_list[1:], {
                "default": "en"
            })
        }}

    CATEGORY = "ATOM2UEKI_WhisperX"
    RETURN_TYPES = ("SRT", "SRT")
    RETURN_NAMES = ("ori_SRT", "trans_SRT")
    FUNCTION = "get_srt"

    def validate_auth_token(self, use_auth_token):
        if not use_auth_token or use_auth_token.strip() == "":
            raise ValueError("Please provide a valid Hugging Face authentication token. Visit https://hf.co/settings/tokens to create one.")
        return use_auth_token.strip()

    def get_srt(self, audio, model_type, batch_size, if_multiple_speaker,
                use_auth_token, source_language, if_translate, translator, to_language):
        compute_type = "float16"
        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_supported() else "cpu"

        # Validate auth token before proceeding
        if if_multiple_speaker:
            use_auth_token = self.validate_auth_token(use_auth_token)

        try:
            # 1. Transcribe with original whisper (batched)
            model = whisperx.load_model(model_type, device, compute_type=compute_type)
            audio_data = whisperx.load_audio(audio)
            
            # Handle language detection/selection
            if source_language == "auto":
                result = model.transcribe(audio_data, batch_size=batch_size)
                detected_language = result["language"]
            else:
                result = model.transcribe(audio_data, batch_size=batch_size, language=source_language)
                detected_language = source_language

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
            except Exception as e:
                print(f"Warning: Alignment failed - {str(e)}")
                print("Continuing with unaligned transcription...")

            # Clean up GPU memory
            del model_a, model
            torch.cuda.empty_cache()
            gc.collect()

            if if_multiple_speaker:
                try:
                    # Initialize diarization pipeline with explicit model name
                    diarize_model = whisperx.DiarizationPipeline(
                        model_name="pyannote/speaker-diarization@2.1",
                        use_auth_token=use_auth_token,
                        device=device
                    )

                    # Perform diarization
                    diarize_segments = diarize_model(audio_data)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # Clean up
                    del diarize_model
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"Speaker diarization failed: {str(e)}")
                    print("Continuing without speaker labels...")

            # Generate output paths
            timestamp = time.time()
            srt_path = os.path.join(out_path, f"{timestamp}_{base_name}.srt")
            trans_srt_path = os.path.join(out_path, f"{timestamp}_{base_name}_{to_language}.srt")

            # Process segments
            srt_line = []
            trans_srt_line = []
            
            for i, res in enumerate(tqdm(result["segments"], desc="Processing segments...", total=len(result["segments"]))):
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
                    except Exception as e:
                        print(f"Translation failed for segment {i+1}: {str(e)}")
                        trans_srt_line.append(srt_line[-1])

            # Write output files
            with open(srt_path, 'w', encoding="utf-8") as f:
                f.write(srt.compose(srt_line))

            if if_translate:
                with open(trans_srt_path, 'w', encoding="utf-8") as f:
                    f.write(srt.compose(trans_srt_line))
            else:
                trans_srt_path = srt_path

            return (srt_path, trans_srt_path)

        except Exception as e:
            raise RuntimeError(f"Error in WhisperX processing: {str(e)}")
        
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3","distil-large-v3","large-v2"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
        'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
        'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
        'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
        'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
        'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
        'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
        'yeekit', 'youdao']
        lang_list = ["zh","en","ja","ko","ru","fr","de","es","pt","it","ar"]
        return {"required":
                    {"audio": ("AUDIOPATH",),
                     "model_type":(model_list,{
                         "default": "large-v3"
                     }),
                     "batch_size":("INT",{
                         "default": 4
                     }),
                     "if_mutiple_speaker":("BOOLEAN",{
                         "default": False
                     }),
                     "use_auth_token":("STRING",{
                         "default": "put your huggingface user auth token here for Assign speaker labels"
                     }),
                     "if_translate":("BOOLEAN",{
                         "default": False
                     }),
                     "translator":(translator_list,{
                         "default": "alibaba"
                     }),
                     "to_language":(lang_list,{
                         "default": "en"
                     })
                     },
                }

    CATEGORY = "ATOM2UEKI_WhisperX"

    RETURN_TYPES = ("SRT","SRT")
    RETURN_NAMES = ("ori_SRT","trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(self, audio,model_type,batch_size,if_mutiple_speaker,
                use_auth_token,if_translate,translator,to_language):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model(model_type, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment
        language_code=result["language"]
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # print(result["segments"]) # after alignment
        
        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a,model
        if if_mutiple_speaker:
            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            import gc; gc.collect(); torch.cuda.empty_cache(); del diarize_model
        # print(diarize_segments)
        # print(result.segments) # segments are now assigned speaker IDs
        
        srt_path = os.path.join(out_path,f"{time.time()}_{base_name}.srt")
        trans_srt_path = os.path.join(out_path,f"{time.time()}_{base_name}_{to_language}.srt")
        srt_line = []
        trans_srt_line = []
        for i, res in enumerate(tqdm(result["segments"],desc="Transcribing ...", total=len(result["segments"]))):
            start = timedelta(seconds=res['start'])
            end = timedelta(seconds=res['end'])
            try:
                speaker_name = res["speaker"][-1]
            except:
                speaker_name = "0"
            content = res['text']
            srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name+content))
            if if_translate:
                #if i== 0:
                   # _ = ts.preaccelerate_and_speedtest() 
                content = ts.translate_text(query_text=content, translator=translator,to_language=to_language)
                trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=speaker_name+content))
                
        with open(srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(srt_line))
        with open(trans_srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(trans_srt_line))

        if if_translate:
            return (srt_path,trans_srt_path)
        else:
            return (srt_path,srt_path)

class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "_WhisperX"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)
