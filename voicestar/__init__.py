"""
VoiceStar: Robust, Duration-Controllable TTS that can Extrapolate

GitHub: https://github.com/jasonppy/VoiceStar
License: MIT

Copyright (c) 2025 Puyuan Peng
"""

__version__ = "0.1.0"

class VoiceStar:
    """
    VoiceStar API - Easy-to-use Python API for VoiceStar model.
    
    This class provides an easy-to-use interface to the VoiceStar TTS model,
    allowing you to generate speech in the voice of a reference audio sample.
    
    Example:
        ```python
        from voicestar import VoiceStar
        
        # Initialize the model (downloads from HF Hub if needed)
        tts = VoiceStar(model_name="VoiceStar_840M_30s")
        
        # Generate speech in the voice of the reference audio
        audio = tts.generate(
            reference_speech="path/to/reference.wav",
            text="This is the text I want to synthesize.",
            target_duration=5.0  # Optional: specify desired duration in seconds
        )
        
        # Save the generated audio
        audio.save("output.wav")
        ```
    """
    
    def __init__(
        self, 
        model_name="VoiceStar_840M_30s", 
        device=None,
        top_k=10,
        top_p=1.0,
        temperature=1.0,
        repeat_prompt=1,
    ):
        """
        Initialize the VoiceStar TTS model.
        
        Args:
            model_name (str): Model name to use. Options:
                - "VoiceStar_840M_30s" (default): 840M parameter model that can generate up to 30s
                - "VoiceStar_840M_40s": 840M parameter model that can generate up to 40s
            device (str, optional): Device to run inference on. If None, will use CUDA if available,
                then MPS (for Apple Silicon), then CPU.
            top_k (int): Top-k sampling parameter. Higher values = more diversity.
            top_p (float): Top-p sampling parameter (nucleus sampling).
            temperature (float): Sampling temperature. Higher values = more diversity.
            repeat_prompt (int): Number of times to repeat the prompt to improve speaker similarity.
        """
        import os
        import torch
        from huggingface_hub import hf_hub_download
        from argparse import Namespace
        import voicestar.voicestar as voice_star
        from data.tokenizer import AudioTokenizer, TextTokenizer
        from voicestar.api import VoiceStarAPI
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # Download and load model
        torch.serialization.add_safe_globals([Namespace])
        ckpt_fn = hf_hub_download(repo_id="pyp1/VoiceStar", filename=f"{model_name}.pth")
        
        bundle = torch.load(ckpt_fn, map_location=self.device, weights_only=True)
        self.args = bundle["args"]
        self.phn2num = bundle["phn2num"]
        self.model = voice_star.VoiceStarModel(self.args)
        self.model.load_state_dict(bundle["model"])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizers
        if self.args.n_codebooks == 4:
            signature = hf_hub_download(repo_id="pyp1/VoiceCraft", filename="encodec_4cb2048_giga.th")
        elif self.args.n_codebooks == 8:
            signature = hf_hub_download(repo_id="pyp1/VoiceCraft", filename="encodec_8cb1024_giga.th")
        else:
            raise ValueError(f"Invalid number of codebooks: {self.args.n_codebooks}")
            
        self.audio_tokenizer = AudioTokenizer(signature=signature)
        self.text_tokenizer = TextTokenizer(backend="espeak")
        
        # Create API instance
        self.api = VoiceStarAPI(
            model=self.model,
            model_args=self.args,
            phn2num=self.phn2num,
            text_tokenizer=self.text_tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            device=self.device,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        
        # Store parameters
        self.repeat_prompt = repeat_prompt
        
    def generate(
        self, 
        reference_speech, 
        text, 
        reference_text=None,
        target_duration=None,
        output_path=None,
    ):
        """
        Generate speech in the voice of the reference audio.
        
        Args:
            reference_speech (str): Path to reference speech audio file
            text (str): Text to synthesize
            reference_text (str, optional): Reference text transcript. If None, will use
                Whisper to automatically transcribe the reference speech.
            target_duration (float, optional): Target duration in seconds. If None,
                will estimate based on reference speech and text length.
            output_path (str, optional): If provided, saves the generated audio to this path
            
        Returns:
            AudioSegment: The generated audio (can be saved with .save() method)
        """
        import os
        import torch
        import torchaudio
        import whisper
        from voicestar.utils import estimate_duration
        
        # Transcribe reference speech if needed
        if reference_text is None:
            print("[Info] No reference_text provided, transcribing reference_speech with Whisper.")
            wh_model = whisper.load_model("large-v3-turbo")
            result = wh_model.transcribe(reference_speech)
            reference_text = result["text"]
            print(f"[Info] Whisper transcribed text: {reference_text}")
        
        # Estimate duration if not provided
        if target_duration is None:
            target_duration = estimate_duration(reference_speech, text)
            print(f"[Info] Estimated target duration: {target_duration:.2f} seconds")
        
        # Get audio info for prompt_end_frame
        info = torchaudio.info(reference_speech)
        prompt_end_frame = int(100 * info.sample_rate)  # 100 seconds max
        
        # Set delay pattern increment
        delay_pattern_increment = self.args.n_codebooks + 1
        
        # Generate audio
        _, generated_audio = self.api.generate(
            audio_fn=reference_speech,
            target_text=text,
            prompt_end_frame=prompt_end_frame,
            target_generation_length=target_duration,
            delay_pattern_increment=delay_pattern_increment,
            prefix_transcript=reference_text,
            repeat_prompt=self.repeat_prompt,
        )
        
        # Save if output_path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            torchaudio.save(output_path, generated_audio[0].cpu(), 16000)
            print(f"[Success] Generated audio saved to {output_path}")
        
        # Return audio segment for further manipulation
        from dataclasses import dataclass
        
        @dataclass
        class AudioSegment:
            waveform: torch.Tensor
            sample_rate: int = 16000
            
            def save(self, path):
                """Save audio to file"""
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                torchaudio.save(path, self.waveform.cpu(), self.sample_rate)
                return path
                
            def play(self):
                """Play audio (if in notebook environment)"""
                try:
                    from IPython.display import Audio, display
                    display(Audio(self.waveform.cpu().numpy().T, rate=self.sample_rate))
                except ImportError:
                    print("Audio playback requires IPython. Use .save() method instead.")
        
        return AudioSegment(generated_audio[0].cpu())