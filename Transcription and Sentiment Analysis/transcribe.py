import whisper  # Imports the Whisper library for audio transcription.


def transcribe_audio(audio_file_path):  # Defines the transcribe_audio function that takes an audio file path as an argument.
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    transcription = model.transcribe(audio_file_path)

    return transcription
