import subprocess, sys, os, json
import pandas as pd
from datetime import datetime, timedelta

from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000


class Transcriber:
    def __init__(self, model_path):
        self.model = Model(model_path)

    def fmt(self, data):
        data = json.loads(data)
        start = min(r["start"] for r in data.get("result", [{ "start": 0 }]))
        end = max(r["end"] for r in data.get("result", [{ "end": 0 }]))
        return {
            "start": str(timedelta(seconds=start)), 
            "end": str(timedelta(seconds=end)), 
            "text": data["text"]
        }

    def transcribe(self, filename):
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        transcription = []
        ffmpeg_command = [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "quiet",
                "-i",
                filename,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ]
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE) as process:
            start_time = datetime.now() 
            while True:
                data = process.stdout.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    transcription.append(self.fmt(rec.Result()))
            transcription.append(self.fmt(rec.FinalResult()))
            end_time = datetime.now()
            time_elapsed = end_time - start_time
            print(f"Time elapsed  {time_elapsed}")

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_time": time_elapsed,
            "transcription": transcription,
        }

    def convert_to_finder_df(self, filename):

        transcription = self.transcribe(filename)

        df = pd.DataFrame(transcription['transcription'])
        df.to_csv('df_tofind.csv')
        return df


if __name__ == "__main__":

    filename = "media/result.mp3"
    model_path = "vosk-model-en-us-daanzu-20200905"
    transcriber = Transcriber(model_path)
    transcriber.convert_to_finder_df(filename)

