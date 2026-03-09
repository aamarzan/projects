import os
import cv2
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS

class VideoDubber:
    def __init__(self):
        # Initialize Whisper (speech-to-text)
        self.whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    def extract_audio(self, video_path):
        """Extract audio using PyDub"""
        video = AudioSegment.from_file(video_path, format="mp4")
        audio_path = "temp_audio.wav"
        video.export(audio_path, format="wav")
        return audio_path

    def transcribe_audio(self, audio_path):
        """Convert speech to text"""
        result = self.whisper(audio_path)
        return result["text"]

    def translate_text(self, text, target_lang='es'):
        """Translate text"""
        return GoogleTranslator(source='auto', target=target_lang).translate(text)

    def synthesize_speech(self, text, lang, output_file="dubbed_audio.mp3"):
        """Generate TTS audio"""
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        return output_file

    def merge_audio_video(self, video_path, audio_path, output_path):
        """Merge using FFmpeg (via PyDub)"""
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Write video frames (no audio)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp_video.mp4", fourcc, fps, (width, height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)

        video.release()
        out.release()

        # Merge audio using FFmpeg
        os.system(f'ffmpeg -y -i "temp_video.mp4" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{output_path}"')

        # Cleanup
        os.remove("temp_video.mp4")

    def dub_video(self, input_video, output_video, target_lang):
        print("[1/4] Extracting audio...")
        audio_path = self.extract_audio(input_video)

        print("[2/4] Transcribing speech...")
        original_text = self.transcribe_audio(audio_path)

        print(f"[3/4] Translating to {target_lang}...")
        translated_text = self.translate_text(original_text, target_lang)

        print("[4/4] Generating dubbed audio...")
        dubbed_audio = self.synthesize_speech(translated_text, target_lang)

        print("Merging audio with video...")
        self.merge_audio_video(input_video, dubbed_audio, output_video)

        # Cleanup
        os.remove(audio_path)
        os.remove(dubbed_audio)
        print(f"✅ Dubbed video saved to {output_video}")

if __name__ == "__main__":
    dubber = VideoDubber()
    dubber.dub_video("input.mp4", "output.mp4", "es")  # Target language (e.g., Spanish 'es')