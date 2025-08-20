import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import subprocess
import tempfile
import uuid
import shutil
import torch

from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AudioTranscript:
    text: str
    segments: List[Dict[str, Any]]
    language: str


class AudioProcessor:
    def __init__(
        self,
        language: Optional[str] = None,
        model_size_or_path: str = "medium",
        device: str = "cpu",
        cpu_threads: int = 4,
        num_workers: int = 1,
    ):
        """
        Initialize audio processor with specified Whisper model size or model path.
        Defaults to the 'medium' model. Uses CPU by default.
        """
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except Exception as e:
            logger.error("faster-whisper is required: pip install faster-whisper\n%s", e)
            raise

        # Cache dir log (for visibility)
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        logger.debug("Using HuggingFace cache directory: %s", cache_dir)

        # Configure threads on CPU
        self.device = device
        if self.device == "cpu":
            try:
                torch.set_num_threads(max(1, cpu_threads))
            except Exception:
                pass

        # Compute type selection
        # - CPU: float32 (stable)
        # - CUDA: float16 (faster), fall back to float32 if not supported
        compute_type = "float32"
        if self.device == "cuda":
            compute_type = "float16"

        self.language = language if language else None

        # Check ffmpeg/ffprobe once
        self.has_ffmpeg = _binary_exists("ffmpeg")
        self.has_ffprobe = _binary_exists("ffprobe")
        if not self.has_ffmpeg:
            logger.warning("FFmpeg not found. Please install ffmpeg for audio extraction.")
        if not self.has_ffprobe:
            logger.debug("ffprobe not found. Will rely on ffmpeg/pydub errors to detect audio streams.")

        # Load model
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                model_size_or_path,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            logger.info(
                "Whisper loaded: model=%s device=%s compute_type=%s language=%s",
                model_size_or_path, self.device, compute_type, self.language or "auto"
            )
        except Exception as e:
            logger.error("Error loading Whisper model: %s", e)
            raise

    def extract_audio(self, video_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Extract audio as 16kHz mono WAV for Whisper.
        Returns None if the video has no audio streams.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"audio-{uuid.uuid4().hex}.wav"

        # Quick pre-check with ffprobe (if available)
        if self.has_ffprobe:
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(video_path)
                    ],
                    capture_output=True, text=True, check=True
                )
                if not result.stdout.strip():
                    logger.info("No audio streams found (ffprobe). Skipping extraction.")
                    return None
            except subprocess.CalledProcessError as e:
                logger.debug("ffprobe failed (continuing to ffmpeg): %s", e)

        # Extract with ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-y", str(audio_path)
                ],
                check=True, capture_output=True
            )
            logger.debug("Audio extracted via ffmpeg → %s", audio_path)
            return audio_path

        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode() if e.stderr else str(e)
            logger.warning("ffmpeg extraction error: %s", error_output)

            # If clearly no audio, return None
            msg = error_output.lower()
            if any(kw in msg for kw in [
                "does not contain any stream",
                "stream map '0:a",
                "output file #0 does not contain any stream",
                "no audio stream"
            ]):
                logger.info("No audio detected (ffmpeg).")
                return None

            # Fallback: pydub (still requires ffmpeg/avlib underneath)
            try:
                logger.info("Falling back to pydub for audio extraction…")
                audio_seg = AudioSegment.from_file(str(video_path))
                audio = audio_seg.set_channels(1).set_frame_rate(16000)
                audio.export(str(audio_path), format="wav")
                logger.debug("Audio extracted via pydub → %s", audio_path)
                return audio_path
            except Exception as e2:
                logger.error("pydub extraction failed: %s", e2)
                raise RuntimeError(
                    "Failed to extract audio. Please install ffmpeg.\n"
                    "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
                    "macOS: brew install ffmpeg\n"
                    "Windows: choco install ffmpeg"
                ) from e2

    def transcribe(self, audio_path: Path) -> Optional[AudioTranscript]:
        """Transcribe audio with faster-whisper (VAD on, beam search, word timestamps)."""
        accepted_languages = {
            "af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el",
            "en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy",
            "id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi",
            "mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru",
            "sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl",
            "tr","tt","uk","ur","uz","vi","yi","yo","zh","yue"
        }
        lang = self.language if (self.language and self.language in accepted_languages) else None
        if self.language and self.language not in accepted_languages:
            logger.warning("Invalid language code '%s'; falling back to auto-detect.", self.language)

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                language=lang,
            )

            segments_list = list(segments)
            if not segments_list:
                logger.warning("No speech detected in audio.")
                return None

            segment_data: List[Dict[str, Any]] = []
            for seg in segments_list:
                seg_words = []
                for w in (seg.words or []):
                    seg_words.append(
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    )
                segment_data.append(
                    {"text": seg.text, "start": seg.start, "end": seg.end, "words": seg_words}
                )

            full_text = " ".join(s.text for s in segments_list).strip()
            return AudioTranscript(text=full_text, segments=segment_data, language=info.language)

        except Exception as e:
            logger.error("Error transcribing audio: %s", e, exc_info=True)
            return None


def _binary_exists(name: str) -> bool:
    """Return True if executable is available on PATH."""
    return shutil.which(name) is not None