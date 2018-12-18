import librosa.display
import scipy.signal.windows
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


class FeatureExtraction:
    def __init__(self):
        self.frame_size = None
        self.overlap = None
        self.sr = None
        self.n_mels = 40
        self.n_MFCCs = 40
        self.base_path = None
        self.results = None
        self.visualise = False
        print("Feature extraction object created")

    def extract_features(self,
                         base_dir,
                         results_path,
                         sampling_rate,
                         framesize,
                         frameoverlap,
                         num_mels = None,
                         num_MFCCs = None,
                         VISUALISE=False):
        # ------------ Extraction Settings ------------
        self.base_path = base_dir
        self.results = results_path
        self.sr = sampling_rate
        self.frame_size = framesize
        self.overlap = frameoverlap
        if num_mels is None:
            n_mels = self.n_mels
        else:
            n_mels = num_mels
        if num_MFCCs is None:
            n_MFCCs = self.n_MFCCs
        else:
            n_MFCCs = num_MFCCs
        if VISUALISE is not False:
            self.visualise = True
        # ---------- Generating file lists ----------
        f_file_list = glob.glob(self.base_path + "*/F*/*.wav")
        m_f_file_list = glob.glob(self.base_path + "*/M*/*.wav")
        # ------------- Processing files -------------
        for file in range(len(f_file_list)):
            self.process_file(f_file_list[file], self.sr, "Female")
            print("Female audio features extracted.")
        for file in range(len(m_f_file_list)):
            self.process_file(m_f_file_list[file], self.sr, "Male")
            print("Male audio features extracted.")
        print("Feature extraction is complete.")

    def extract_file_features(self,
                              file_path,
                              sr = None,
                              VISUALISE=False):
        # ------------ Extraction Settings ------------
        if VISUALISE is not False:
            self.visualise = True
        # ------------- Processing files -------------
        file_results_path = self.process_single_file(results_path= "results/", path_to_file=file_path, sr=self.sr)
        print("Feature extraction is complete.")
        return file_results_path

    def process_single_file(self, path_to_file, results_path, sr):
        filename_with_ext = os.path.basename(path_to_file)
        file_base = path_to_file.replace(self.base_path, "")
        base_nofile = file_base.replace(filename_with_ext, "")
        result_loc = self.results + base_nofile
        filename_with_ext = filename_with_ext.replace(".wav", ".csv")

        data, sr = librosa.load(path_to_file, sr=sr)  # returns ndarray of floating point numbers

        data_length = len(data)
        window_length = self.frame_size  # length in samples
        window_shift = int(self.overlap * window_length)
        total_windows = int(data_length / window_shift)

        power_spectrum = np.abs(librosa.stft(data,
                                             hop_length=window_shift,
                                             n_fft=window_length,
                                             center=True,
                                             window=(scipy.signal.get_window('hann', window_length)),
                                             pad_mode='edge')) ** 2
        mel_power_spectrum = librosa.feature.melspectrogram(S=power_spectrum,
                                                            n_fft=window_length,
                                                            hop_length=window_shift,
                                                            n_mels = self.n_mels)
        log_power_mel_spectrogram = librosa.power_to_db(mel_power_spectrum,
                                                        ref=np.max)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_power_spectrum),
                                     n_mfcc=self.n_mels, # calculat all, discard everything outside 2-13
                                     sr=sr)
        # mfccs = mfccs[:, 1:(n_mfcc+1)]

        if self.visualise:  # VISUALISE DATA (for development purposes)
            plt.figure()
            plt.subplot(3, 1, 1)
            librosa.display.waveplot(y=data, sr=sr, x_axis='time')
            plt.title('Sound Signal')
            plt.subplot(3, 1, 2)
            plt.subplot(3, 1, 3)
            librosa.display.specshow(power_spectrum, sr=sr, hop_length=window_shift, y_axis='log')
            plt.colorbar()
            plt.title('Power spectrum')
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(power_spectrum, ref=np.max), sr=sr, y_axis='log', x_axis='time',
                                     hop_length=window_shift)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-Power spectrogram')
            plt.figure(figsize=(10,4))
            librosa.display.specshow(mel_power_spectrum, sr=sr, y_axis='mel', x_axis='time', hop_length=window_shift)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Power spectrogram')
            librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', sr=sr, hop_length=window_shift)
            plt.colorbar()
            plt.title('MFCCs')
            plt.tight_layout()
            plt.show()
        else:
            mfccs = mfccs.T                     # transpose MFFCs so that they are columns, rows are frames
            gender_arr = np.zeros((mfccs.shape[0],1))           # add column showing gender, 0 is female and 1 is male
            gender_arr = gender_arr.reshape(mfccs.shape[0],1)
            mfccs = np.hstack([mfccs, gender_arr])              # combine MFCC and gender column
            if not os.path.isdir(result_loc):
                os.makedirs(result_loc)
            np.savetxt(result_loc + filename_with_ext, mfccs, delimiter=',')  # output to filename.csv
        return result_loc + filename_with_ext

    def process_file(self, path_to_file, sr, gender):
        filename_with_ext = os.path.basename(path_to_file)
        file_base = path_to_file.replace(self.base_path, "")
        base_nofile = file_base.replace(filename_with_ext, "")
        result_loc = self.results + base_nofile
        filename_with_ext = filename_with_ext.replace(".wav", ".csv")

        data, sr = librosa.load(path_to_file, sr=sr)  # returns ndarray of floating point numbers

        data_length = len(data)
        window_length = self.frame_size  # length in samples
        window_shift = int(self.overlap * window_length)
        total_windows = int(data_length / window_shift)

        power_spectrum = np.abs(librosa.stft(data,
                                             hop_length=window_shift,
                                             n_fft=window_length,
                                             center=True,
                                             window=(scipy.signal.get_window('hamming', window_length)),
                                             pad_mode='edge')) ** 2
        mel_power_spectrum = librosa.feature.melspectrogram(S=power_spectrum,
                                                            n_fft=window_length,
                                                            hop_length=window_shift,
                                                            n_mels = self.n_mels)
        log_power_mel_spectrogram = librosa.power_to_db(mel_power_spectrum,
                                                        ref=np.max)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_power_spectrum),
                                     n_mfcc=self.n_mels, # calculat all, discard everything outside 2-13
                                     sr=sr)
        # mfccs = mfccs[:, 1:(n_mfcc+1)]

        if self.visualise:  # VISUALISE DATA (for development purposes)
            plt.figure()
            plt.subplot(3, 1, 1)
            librosa.display.specshow(power_spectrum, sr=sr, hop_length=window_shift, y_axis='log')
            plt.colorbar()
            plt.title('Power spectrogram')
            plt.subplot(3, 1, 2)
            librosa.display.specshow(librosa.power_to_db(power_spectrum, ref=np.max), sr=sr, y_axis='log', x_axis='time',
                                     hop_length=window_shift)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-Power spectrogram')
            plt.subplot(3, 1, 3)
            librosa.display.specshow(log_power_mel_spectrogram, sr=sr, y_axis='mel', x_axis='time', hop_length=window_shift)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Power spectrogram')
            plt.figure(figsize=(10, 4))
            librosa.display.waveplot(data, sr=sr)
            plt.title('Audio wave plot')
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', sr=sr, hop_length=window_shift)
            plt.colorbar()
            plt.title('MFCCs')
            plt.tight_layout()
            plt.show()
        else:
            mfccs = mfccs.T                     # transpose MFFCs so that they are columns, rows are frames
            if gender is "Female":
                gender_arr = np.zeros((mfccs.shape[0],1))           # add column showing gender, 0 is female and 1 is male
            elif gender is "Male":
                gender_arr = np.ones((mfccs.shape[0],1))
            gender_arr = gender_arr.reshape(mfccs.shape[0],1)
            mfccs = np.hstack([mfccs, gender_arr])              # combine MFCC and gender column
            if not os.path.isdir(result_loc):
                os.makedirs(result_loc)
            np.savetxt(result_loc + filename_with_ext, mfccs, delimiter=',')  # output to filename.csv

    def load_audio(self, path_to_file, sr=44100):
        data, sr = librosa.load(path_to_file, sr)
        return data
