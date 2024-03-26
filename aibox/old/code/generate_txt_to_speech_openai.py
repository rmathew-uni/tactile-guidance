from openai import OpenAI

unique_api_key = "YOUR_OPENAI_API_KEY"

# initialize the OpenAI API client
client = OpenAI(api_key=unique_api_key)

# Sample text to generate speech from
text = "banana"

# The voice to use, there is alloy, echo, fable, onyx, nova, and shimmer
open_api_voice = "nova"

# generate speech from the text
response = client.audio.speech.create(
    model="tts-1-hd", # the model to use: tts-1 and tts-1-hd
    voice=open_api_voice, 
    input=text,
    speed=1.0, # ranging from 0.25 to 4.0
)
# save the generated speech to a file
response.stream_to_file(f"{text}_{open_api_voice}.mp3")
