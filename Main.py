#   Gender Classification System for Final Year Project Stage 1
#
#   Paveen Rai
#   1561451
#

# Library imports
from NeuralNetwork import NeuralNetwork
from Spectrogram import FeatureExtraction
import IPython.display as ipd
import simpleaudio as sa
import glob
import tkinter as tk
from tkinter import ttk
from tkinter import Menu
from tkinter import messagebox

# Program Setup
extract_features = False
base_dir = "../../Sounds/Speech_TIMIT/"
results_dir = "../../RESULTS/"
test_sound_file = ""
frame_size = 512
frame_overlap = 0.5 # as a decimal
sampling_rate = 44100

feature_extraction = FeatureExtraction()
# Create Feature Extraction object & begin
if extract_features is True:
    feature_extraction.extract_features(framesize=frame_size,
                                        frameoverlap=frame_overlap,
                                        base_dir=base_dir,
                                        results_path=results_dir,
                                        sampling_rate=sampling_rate)

# Create NN
nn = NeuralNetwork()
nn.train(results_loc=results_dir)
nn.test()

input_sound = input("Would you like to test a single audio file?")

if input_sound == "Yes":
    test_sound_files = "../../Sounds/Speech_TIMIT/test/MJWT0/SA1.wav"
    test_csv_file = "../../RESULTS/test/MJWT0/SA1.csv"
    prediction = nn.predict_file(test_csv_file)
    if prediction is 0:
        prediction = "Female"
    else:
        prediction = "Male"
    print("Prediction: ", prediction)
    # display result
    # play audio
    wav_object = sa.WaveObject.from_wave_file(test_sound_files)
    play_obj = wav_object.play()
    play_obj.wait_done()
# Female voice
    test_sound_files = "../../Sounds/Speech_TIMIT/test/FKFB0/SA1.wav"
    test_csv_file = "../../RESULTS/test/FKFB0/SA1.csv"
    prediction = nn.predict_file(test_csv_file)
    if prediction is 0:
        prediction = "Female"
    else:
        prediction = "Male"
    print("Prediction: ", prediction)
    # display result
    # play audio
    wav_object = sa.WaveObject.from_wave_file(test_sound_files)
    play_obj = wav_object.play()
    play_obj.wait_done()

print("End of program")

