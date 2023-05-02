import os
import time
import streamlit as st
import io
import moviepy.editor as mp
from test_simple import Transcriber
from pinecone_sentence_transformers import get_query

transcriber = Transcriber('vosk-model-en-us-daanzu-20200905')

st.title("Play Uploaded File")
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
temporary_location = False

if uploaded_file is not None:
    g = io.BytesIO(uploaded_file.read())
    temporary_video_file = "media/testout_simple.mp4"
    with open(temporary_video_file, 'wb') as out:
        out.write(g.read())
    out.close()

    my_clip = mp.VideoFileClip(temporary_video_file)
    my_clip.audio.write_audiofile(r"media/result.mp3")
    time.sleep(2)
    os.system(f"rm -rf {temporary_video_file}")

st.video(uploaded_file)
transcription_df = transcriber.convert_to_finder_df('media/result.mp3')
transcription_df.to_csv('df_tofind.csv')
if transcription_df is not None:
    st.write('transcription is over')
    question = st.text_input('Write your question on video: ')

    if len(question) > 1:
        timestamps = get_query(question)
        st.write('Timestamps on video: ', timestamps)
        total_sec = timestamps['time_start'].minute*60 + timestamps['time_start'].second
        st.video(uploaded_file, start_time=total_sec)

# st.dataframe(transcription_df)


