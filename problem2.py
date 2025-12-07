import pyttsx3
engine = pyttsx3.init()
# For Mac, If you face error related to "pyobjc" when running the `init()` method :
# Install 9.0.1 version of pyobjc : "pip install pyobjc>=9.0.1"
engine.say("hello friends i am america here to meet you all and so exited to be part of this show thank you for calling me")
engine.runAndWait()