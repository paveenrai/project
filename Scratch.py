from Spectrogram import FeatureExtraction

fe = FeatureExtraction()

fe.extract_file_features(file_path="../../Sounds/Speech_TIMIT/dev/FETB0/SA1.wav", VISUALISE=True, sr=44100)