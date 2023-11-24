
import cv2
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pickle
import shutil
from tensorflow.keras import models
from PIL import Image
from python_speech_features.sigproc import framesig
from python_speech_features.sigproc import logpowspec
from python_speech_features.sigproc import preemphasis
from scipy.signal.windows import hamming
from PyQt5 import QtCore, uic, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from scipy.fftpack import dct, idct
from scipy.signal import medfilt
import tensorflow as tf
import scipy.signal as sig
import librosa
import pygame
import soundfile as sf
from tensorflow.keras import backend as k_backend
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, LeakyReLU


def message_info(message):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(message)
    msg.setWindowTitle(u"Информация")
    msg.show()
    msg.exec_()


def phase_spectrogram(filename, progress_Bar):
    fft = 512
    win_len = 400
    step = 160
    try:
        rate, audio = wav.read(os.path.join('App', 'temp', filename))
        audio_psp = audio
        audio_frame_cepstr = framesig(audio_psp, win_len, step, winfunc=hamming)
        audio = audio / 32768
        audio = preemphasis(audio, coeff=0.97)
        audio = framesig(audio, win_len, step, winfunc=hamming)
        progress_Bar.setValue(5)
        phasespec_instantaneous_frequency_derivative = np.diff(np.angle(np.fft.rfft(audio, fft)).T)
        step_p = (20 / phasespec_instantaneous_frequency_derivative.shape[0])
        new_value = progress_Bar.value() + step_p
        for i in range(phasespec_instantaneous_frequency_derivative.shape[0]):
            new_value = new_value + step_p
            progress_Bar.setValue(round(new_value))
            for j in range(phasespec_instantaneous_frequency_derivative.shape[1]):
                if phasespec_instantaneous_frequency_derivative[i, j] > np.pi:
                    phasespec_instantaneous_frequency_derivative[i, j] = \
                        phasespec_instantaneous_frequency_derivative[i, j] - 2 * np.pi
                if phasespec_instantaneous_frequency_derivative[i, j] < -np.pi:
                    phasespec_instantaneous_frequency_derivative[i, j] = \
                        phasespec_instantaneous_frequency_derivative[i, j] + 2 * np.pi
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_if.png'),
                   phasespec_instantaneous_frequency_derivative, cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_if.pickle'), 'wb') as f:
            pickle.dump(phasespec_instantaneous_frequency_derivative, f)

        progress_Bar.setValue(25)
        step_p = (25 / phasespec_instantaneous_frequency_derivative.shape[0])
        new_value = progress_Bar.value() + step_p
        basehand_phase_difference = np.zeros(phasespec_instantaneous_frequency_derivative.shape)
        for i in range(phasespec_instantaneous_frequency_derivative.shape[0]):
            new_value = new_value + step_p
            progress_Bar.setValue(round(new_value))
            for j in range(phasespec_instantaneous_frequency_derivative.shape[1]):
                basehand_phase_difference[i, j] = phasespec_instantaneous_frequency_derivative[i, j] - \
                                                  (2 * np.pi * i / fft) * step
                if basehand_phase_difference[i, j] > np.pi:
                    while basehand_phase_difference[i, j] > np.pi:
                        basehand_phase_difference[i, j] = basehand_phase_difference[i, j] - 2 * np.pi
                if basehand_phase_difference[i, j] < -np.pi:
                    while basehand_phase_difference[i, j] < -np.pi:
                        basehand_phase_difference[i, j] = basehand_phase_difference[i, j] + 2 * np.pi
        progress_Bar.setValue(50)
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_bpd.png'),
                   basehand_phase_difference[0:256, :], cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_bpd.pickle'), 'wb') as f:
            pickle.dump(basehand_phase_difference[0:256, :], f)
        progress_Bar.setValue(55)

        ro = 1.0
        alpha = 0.6
        audio_nx = np.multiply(audio, range(1, audio.shape[1] + 1))
        complex_spec = np.fft.rfft(audio, fft)
        complex_spec_nx = np.fft.rfft(audio_nx, fft)
        smoof_spec_medfilt = medfilt(np.log((np.abs(complex_spec) / fft) + 0.0000001), kernel_size=(5, 1))
        progress_Bar.setValue(60)
        smoof_spec_dct = dct(smoof_spec_medfilt, type=2, norm='ortho', n=round(fft / 2 + 1))
        smoof_spec_dct[:, 30:] = 0
        smoof_spec = idct(smoof_spec_dct, type=2, norm='ortho', n=round(fft / 2 + 1))
        tau = ((np.real(complex_spec) * np.real(complex_spec_nx)) + (
                    np.imag(complex_spec_nx) * np.imag(complex_spec))) / np.exp(smoof_spec) ** (2 * ro)

        tau_group = tau * (np.abs(tau) ** (alpha - 1))
        tau_group = np.log10(np.absolute(tau_group))
        tau_group[np.isnan(tau_group)] = 90
        progress_Bar.setValue(65)
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_mgd.png'),
                   tau_group.T, cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_mgd.pickle'), 'wb') as f:
            pickle.dump(tau_group.T, f)
        progress_Bar.setValue(70)

        logspec = logpowspec(audio, fft, norm=1)
        # fig, ax = plt.subplots()
        # im = ax.imshow(logspec.T, cmap='jet', origin='lower')
        # fig.colorbar(im)
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_lms.png'), logspec.T,
                   cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_lms.pickle'), 'wb') as f:
            pickle.dump(logspec.T, f)
        progress_Bar.setValue(75)

        phasespec = np.angle(np.fft.rfft(audio, fft))
        phasespec_group_delay = np.diff(phasespec)
        progress_Bar.setValue(80)
        step_p = (10 / phasespec_group_delay.shape[0])
        new_value = progress_Bar.value() + step_p
        for i in range(phasespec_group_delay.shape[0]):
            new_value = new_value + step_p
            progress_Bar.setValue(round(new_value))
            for j in range(phasespec_group_delay.shape[1]):
                if phasespec_group_delay[i, j] > np.pi:
                    while phasespec_group_delay[i, j] > np.pi:
                        phasespec_group_delay[i, j] = phasespec_group_delay[i, j] - 2 * np.pi
                if phasespec_group_delay[i, j] < -np.pi:
                    while phasespec_group_delay[i, j] < -np.pi:
                        phasespec_group_delay[i, j] = phasespec_group_delay[i, j] + 2 * np.pi
        progress_Bar.setValue(90)
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_gd.png'), phasespec_group_delay.T,
                   cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_gd.pickle'), 'wb') as f:
            pickle.dump(phasespec_group_delay.T, f)
        progress_Bar.setValue(92)

        with open(os.path.join('App', 'temp', 'set_main_tone.txt'), 'r') as text_file:
            strings_line = text_file.readlines()
        try:
            height_ = int(strings_line[0].split('\n')[0])
            distance_ = int(strings_line[1])
            metki, xz = sig.find_peaks(audio_psp, height=height_, distance=distance_)
        except ValueError:
            message_info('Ошибка в файле ' + os.path.join('App', 'temp', 'set_main_tone.txt'))

        audio_frame_psp = np.zeros(round(fft / 2) + 1)
        for i in range(metki.shape[0] - 3):
            frame_psp = audio_psp[metki[i]:metki[i + 3]]
            frame_psp = np.angle(np.fft.rfft(frame_psp * np.hamming(frame_psp.shape[0]), fft) / fft)
            audio_frame_psp = np.vstack((audio_frame_psp, frame_psp))
        audio_frame_psp = audio_frame_psp[1:]
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_psp.png'),
                   audio_frame_psp.T, cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_psp.pickle'), 'wb') as f:
            pickle.dump(audio_frame_psp.T, f)
        progress_Bar.setValue(96)

        cepstr = np.log(np.abs(np.fft.rfft(audio_frame_cepstr, fft) ** 2))
        progress_Bar.setValue(99)
        cepstr = np.fft.irfft(cepstr).T
        cepstr = cepstr[round(rate / 500):round(rate / 70), :]
        cepstr = cepstr[::-1, :]
        plt.imsave(os.path.join('App', 'temp', filename.split('.wav')[0] + '_cepstr.png'),
                   np.abs(cepstr), cmap='jet', origin='lower')
        with open(os.path.join('App', 'temp', filename.split('.wav')[0] + '_cepstr.pickle'), 'wb') as f:
            pickle.dump(np.abs(cepstr), f)
        progress_Bar.setValue(100)
    except Exception as e:
        message_info(str(e))


def preprocess(dir_name, file_name):
    try:
        data_, rate_ = librosa.load(os.path.join(dir_name, file_name), sr=16000)
        new_name = file_name.split('.wav')[0] + '_16000' + '.wav'

        if not os.path.exists(os.path.join('App', 'temp')):
            os.makedirs(os.path.join('App', 'temp'))

        new_dir = os.path.join('App', 'temp')

        sf.write(os.path.join('App', 'temp', new_name), data_, 16000)
        return new_dir, new_name
    except Exception as e:
        message_info(str(e))


def open_img(dir_name, file_name):
    img = Image.open(os.path.join(dir_name, file_name.split('.wav')[0] + '.png')).convert('RGB')
    img = crop_image(img, width=224, height=224, seed=3)
    return img


def crop_image(img, width=None, height=None, seed=1):
    np.random.seed(seed)
    x_ = np.random.randint(img.size[0] - height)
    y_ = np.random.randint(img.size[1] - width)
    return img.crop((x_, y_, width + x_, height + y_))


def read_and_transform_ps(dir_name, file_name):
    fft = 512
    win_len = 400
    step = 160
    try:
        rate, audio = wav.read(os.path.join(dir_name, file_name))
        audio = audio / 32768
        audio = preemphasis(audio, coeff=0.97)
        audio = framesig(audio, win_len, step, winfunc=hamming)
        audio_nx = np.multiply(audio, range(1, audio.shape[1] + 1))
        complex_spec = np.fft.rfft(audio, fft)
        complex_spec_nx = np.fft.rfft(audio_nx, fft)
        product_spectrum = np.log(np.abs((np.real(complex_spec) * np.real(complex_spec_nx)) +
                                         (np.imag(complex_spec_nx) * np.imag(complex_spec))) + 0.00001)
        product_spectrum = product_spectrum / (np.max(np.abs(product_spectrum)))

        if product_spectrum.shape[0] < 256:
            product_concat = product_spectrum[::-1]
            while product_spectrum.shape[0] < 256:
                product_spectrum = np.vstack((product_spectrum, product_concat))
                product_concat = product_spectrum[::-1]

            plt.imsave(os.path.join(dir_name, file_name.split('.wav')[0] + '.png'),
                       product_spectrum[0:256, :].T, cmap='jet', origin='lower')
        else:
            plt.imsave(os.path.join(dir_name, file_name.split('.wav')[0] + '.png'),
                       product_spectrum.T, cmap='jet', origin='lower')
    except Exception as e:
        message_info(str(e))


def prediction(creating_model, image):
    x = creating_model.predict(image)
    return x[0][0], x[0][1]


def create_network():
    base_model = efficientnet.EfficientNetB7(include_top=False, input_shape=(224, 224, 3))

    model_new = Sequential()
    model_new.add(base_model)
    model_new.add(Flatten())

    model_new.add(Dense(512))
    model_new.add(BatchNormalization())
    model_new.add(Activation(LeakyReLU(alpha=0.2)))
    # model_new.add(Dropout(0.3))

    model_new.add(Dense(256))
    model_new.add(BatchNormalization())
    model_new.add(Activation(LeakyReLU(alpha=0.2)))
    # model_new.add(Dropout(0.3))

    model_new.add(Dense(128))
    model_new.add(BatchNormalization())
    model_new.add(Activation(LeakyReLU(alpha=0.2)))
    # model_new.add(Dropout(0.3))

    model_new.add(Dense(64))
    model_new.add(BatchNormalization())
    model_new.add(Activation(LeakyReLU(alpha=0.2)))
    # model_new.add(Dropout(0.3))

    model_new.add(Dense(2))
    model_new.add(Activation('softmax'))

    return model_new


class FileAnalisys(QtWidgets.QWidget):
    def __init__(self):
        super(FileAnalisys, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.curdir), 'App', 'form', 'main.ui'), self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.line_filename.setText(u'Аудиофайл не выбран')
        self.create_model = create_network()
        if self.type_detect.currentText() == 'Детектирование синтеза речи':
            self.create_model.load_weights(os.path.join('App', 'model', 'model_weights.h5'))
        elif self.type_detect.currentText() == 'Детектирование модификации речи':
            self.create_model.load_weights(os.path.join('App', 'model', 'model_mod_weights.h5'))
        self.last_conv_layer = self.create_model.get_layer('flatten')
        self.model_inputs = self.create_model.inputs
        self.model_outputs = self.create_model.output
        self.iterate = Model(inputs=[self.model_inputs], outputs=[self.model_outputs, self.last_conv_layer.input])
        self.analisys_file_btn.setVisible(False)
        self.label_ps.setVisible(False)
        self.product_spectrum_img.setVisible(False)
        self.label_ps_crop.setVisible(False)
        self.product_spectrum_img_crop.setVisible(False)
        self.probability_bonafide.setVisible(False)
        self.probability_spoof.setVisible(False)
        self.label_analisys.setVisible(False)
        self.label_result.setVisible(False)
        self.play_audio.setVisible(False)
        self.stop_audio.setVisible(False)
        self.open_phase_spectrs.setVisible(False)
        self.label_ok.setVisible(False)
        self.label_stop.setVisible(False)
        self.label_phase.setVisible(False)
        self.progress_phase.setVisible(False)
        self.open_setting.setVisible(False)
        self.heatmap.setVisible(False)
        self.heatmap_filters.setVisible(False)
        self.type_detect.setVisible(False)

        if not os.path.exists(os.path.join('App', 'temp')):
            os.makedirs(os.path.join('App', 'temp'))

        if os.path.isfile(os.path.join(os.path.abspath(os.curdir), 'App', 'temp', 'directory.txt')) and \
                os.path.getsize(os.path.join(os.path.abspath(os.curdir), 'App', 'temp', 'directory.txt')) > 0:
            with open(os.path.join(os.path.abspath(os.curdir), 'App', 'temp', 'directory.txt'), 'r') as text_file:
                self.directory = text_file.read()
        else:
            try:
                if sys.platform == 'linux':
                    self.directory = '/home'
                else:
                    self.directory = 'C:\\'
            except Exception as e:
                message_info(str(e))

        self.filename_and_dir = ''
        self.filename_audio = ''

        self.openfile.clicked.connect(self.open_file)
        self.analisys_file_btn.clicked.connect(self.prediction_class)
        self.play_audio.clicked.connect(self.play_audio_func)
        self.stop_audio.clicked.connect(self.stop_audio_func)
        self.open_phase_spectrs.clicked.connect(self.open_phase_spectrs_func)
        self.open_setting.clicked.connect(self.open_setting_func)
        self.heatmap.clicked.connect(self.get_heatmap)
        self.heatmap_filters.clicked.connect(self.get_heatmap_filters)
        self.type_detect.currentIndexChanged.connect(self.type_detect_changed)

    def type_detect_changed(self):
        if self.type_detect.currentText() == 'Детектирование синтеза речи':
            self.create_model.load_weights(os.path.join('App', 'model', 'model_weights.h5'))
        elif self.type_detect.currentText() == 'Детектирование модификации речи':
            self.create_model.load_weights(os.path.join('App', 'model', 'model_mod_weights.h5'))
        self.label_ok.setVisible(False)
        self.label_stop.setVisible(False)
        self.label_analisys.setVisible(False)
        self.probability_bonafide.setVisible(False)
        self.probability_spoof.setVisible(False)
        self.label_result.setVisible(False)

        if os.path.exists(os.path.join('App', 'temp', 'filters')):
            shutil.rmtree(os.path.join('App', 'temp', 'filters'))

        list_file = os.listdir(os.path.join('App', 'temp'))

        self.analisys_file_btn.setVisible(True)

    def heatmap_filters_show(self, image):
        if os.path.exists(os.path.join('App', 'temp', 'filters')) and len(os.listdir(
                os.path.join('App', 'temp', 'filters'))) == 2560:
            pictures = os.listdir(os.path.join('App', 'temp', 'filters'))
            pictures = sorted(pictures)
            self.label_analisys.setVisible(False)
            self.h_s = HeatMapsFilters(pictures)
            self.h_s.show()
        else:
            if not os.path.exists(os.path.join('App', 'temp', 'filters')):
                os.makedirs(os.path.join('App', 'temp', 'filters'))
            if os.path.isfile(os.path.join(image)):
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(image.split('.')[0] + '_gray_filt.png', gray)
                image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
                image_tensor = tf.expand_dims(image_tensor, 0)
                preds__ = self.iterate.predict(image_tensor)
                img_ = cv2.imread(image.split('.')[0] + '_gray_filt.png')
                for act in range(0, preds__[1].shape[3]):
                    heatmap = preds__[1][:, :, :, act]
                    heatmap = np.maximum(heatmap, 0)
                    np.seterr(invalid='ignore')
                    heatmap /= np.max(heatmap)
                    heatmap = heatmap.reshape((7, 7))

                    heatmap = cv2.resize(heatmap, (img_.shape[1], img_.shape[0]))

                    heatmap = np.uint8(255 * heatmap)

                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    superimposed_img = heatmap * 0.4 + img_

                    cv2.imwrite(os.path.join('App', 'temp', 'filters', image.split('/')[-1].split('.')[0] +
                                             '_' + str(act) + '.png'), superimposed_img)

                    pictures = os.listdir(os.path.join('App', 'temp', 'filters'))
                    pictures = sorted(pictures)

                self.label_analisys.setVisible(False)
                self.h_s = HeatMapsFilters(pictures)
                self.h_s.show()
            else:
                message_info('Не найден файл: ' + image)

    def get_heatmap_filters(self):
        self.label_analisys.setText('Вычисление карт активации ...')
        self.label_analisys.setVisible(True)
        qApp.processEvents()
        self.heatmap_filters_show(os.path.join(self.name_dir_after_preprocess,
                                               self.name_file_after_preprocess.split('.wav')[0] + '_crop.png'))

    def get_img(self, image_, name__):
        image_read = cv2.imread(image_)
        image_tensor = tf.convert_to_tensor(image_read, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        with tf.GradientTape() as tape:
            model_out, last_conv_layer = self.iterate(image_tensor)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = k_backend.mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((7, 7))

        img = cv2.imread(image_)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_.split('.')[0] + '_gray.png', gray)
        img = cv2.imread(image_.split('.')[0] + '_gray.png')

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        cv2.imwrite(image_.split('.')[0] + name__, superimposed_img)

        plt.figure()
        plt.title('Тепловая карта активации фильтров')
        plt.gcf().canvas.set_window_title(image_.split('/')[-1].split('.')[0].split('_')[0] + '.wav')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(plt.imread(image_.split('.')[0] + name__))
        plt.show()

    def mean_heatmap(self, image_):
        if os.path.isfile(image_):
            if self.type_detect.currentText() == 'Детектирование синтеза речи':
                if os.path.isfile(image_.split('.')[0] + '_gray_heatmap.png'):
                    plt.figure()
                    plt.title('Тепловая карта активации фильтров (среднее значение фильтров)')
                    plt.gcf().canvas.set_window_title(image_.split('/')[-1].split('.')[0].split('_')[0] + '.wav')
                    plt.imshow(plt.imread(image_.split('.')[0] + '_gray_heatmap.png'))
                    plt.show()
                else:
                    self.get_img(image_, '_gray_heatmap.png')
            elif self.type_detect.currentText() == 'Детектирование модификации речи':
                if os.path.isfile(image_.split('.')[0] + '_gray_heatmap_mod.png'):
                    plt.figure()
                    plt.title('Тепловая карта активации фильтров (среднее значение фильтров)')
                    plt.gcf().canvas.set_window_title(image_.split('/')[-1].split('.')[0].split('_')[0] + '.wav')
                    plt.imshow(plt.imread(image_.split('.')[0] + '_gray_heatmap_mod.png'))
                    plt.show()
                else:
                    self.get_img(image_, '_gray_heatmap_mod.png')

        else:
            message_info('Не найден файл: ' + image_)

    def get_heatmap(self):
        self.mean_heatmap(os.path.join(self.name_dir_after_preprocess,
                                       self.name_file_after_preprocess.split('.wav')[0] + '_crop.png'))

    def open_setting_func(self):
        self.settings_windows = SettingMainTone(self.name_file_after_preprocess)
        self.settings_windows.show()

    def open_phase_spectrs_func(self):
        self.set_dis(True)
        qApp.processEvents()
        if os.path.isfile(os.path.join('App', 'temp', 'set_main_tone.txt')):
            if os.path.isfile(os.path.join('App', 'temp',
                                           self.name_file_after_preprocess.split('.wav')[0] + '_bpd.png')) \
                    and os.path.isfile(os.path.join('App', 'temp',
                                                    self.name_file_after_preprocess.split('.wav')[0] + '_mgd.png')) \
                    and os.path.isfile(os.path.join('App', 'temp',
                                                    self.name_file_after_preprocess.split('.wav')[0] + '_if.png')) \
                    and os.path.isfile(os.path.join('App', 'temp',
                                                    self.name_file_after_preprocess.split('.wav')[0] + '_lms.png')) \
                    and os.path.isfile(os.path.join('App', 'temp',
                                                    self.name_file_after_preprocess.split('.wav')[0] + '_gd.png')) \
                    and os.path.isfile(os.path.join('App', 'temp',
                                                    self.name_file_after_preprocess.split('.wav')[0] + '_cepstr.png')):
                with open(os.path.join('App', 'temp', 'set_main_tone.txt'), 'r') as text_file:
                    strings_line = text_file.readlines()
                try:
                    rate, audio = wav.read(os.path.join('App', 'temp', self.name_file_after_preprocess))
                    try:
                        height_ = int(strings_line[0].split('\n')[0])
                        distance_ = int(strings_line[1])
                        metki, xz = sig.find_peaks(audio, height=height_, distance=distance_)
                        audio_frame_psp = np.zeros(round(512 / 2) + 1)
                        for i in range(metki.shape[0] - 3):
                            frame_psp = audio[metki[i]:metki[i + 3]]
                            frame_psp = np.angle(np.fft.rfft(frame_psp * np.hamming(frame_psp.shape[0]), 512) / 512)
                            audio_frame_psp = np.vstack((audio_frame_psp, frame_psp))
                        audio_frame_psp = audio_frame_psp[1:]
                        plt.imsave(os.path.join('App', 'temp', self.name_file_after_preprocess.split('.wav')[0] +
                                                '_psp.png'), audio_frame_psp.T, cmap='jet', origin='lower')
                    except ValueError:
                        message_info('Ошибка в файле ' + os.path.join('App', 'temp', 'set_main_tone.txt'))
                except Exception as e:
                    message_info('Не удаётся прочитать файл' + self.name_file_after_preprocess + '\n' +
                                 str(e))

                self.set_dis(False)
                self.phase_img = PhaseSpectrum(self.name_file_after_preprocess)
                self.phase_img.show()
            else:
                self.label_phase.setVisible(True)
                self.progress_phase.setVisible(True)
                self.progress_phase.setValue(0)
                phase_spectrogram(self.name_file_after_preprocess, self.progress_phase)
                self.set_dis(False)
                self.label_phase.setVisible(False)
                self.progress_phase.setVisible(False)
                self.phase_img = PhaseSpectrum(self.name_file_after_preprocess)
                self.phase_img.show()
        else:
            self.set_dis(False)
            message_info('Настройте частоту основного тона.\n Для этого нажмите кнопку настройки в главном окне')

    def play_audio_func(self):
        self.set_dis(True)
        try:
            pygame.init()
            self.s = pygame.mixer.Sound(os.path.join('App', 'temp', self.name_file_after_preprocess))
            qApp.processEvents()
            self.s.play()
            self.set_dis(False)
        except Exception as e:
            self.set_dis(False)
            message_info(str(e))

    def stop_audio_func(self):
        try:
            self.s.stop()
        except Exception as e:
            message_info(str(e))

        self.set_dis(False)

    def set_dis(self, var_bool):
        self.line_filename.setDisabled(var_bool)
        self.analisys_file_btn.setDisabled(var_bool)
        self.label_ps.setDisabled(var_bool)
        self.product_spectrum_img.setDisabled(var_bool)
        self.label_ps_crop.setDisabled(var_bool)
        self.product_spectrum_img_crop.setDisabled(var_bool)
        self.openfile.setDisabled(var_bool)
        self.label_result.setDisabled(var_bool)
        self.label_ok.setDisabled(var_bool)
        self.label_stop.setDisabled(var_bool)
        self.label_img.setDisabled(var_bool)
        # self.heatmap.setVisible(var_bool)

    def prediction_class(self):
        self.label_analisys.setVisible(True)
        self.analisys_file_btn.setVisible(False)
        self.set_dis(True)
        qApp.processEvents()
        bonafide, spoof = prediction(self.create_model, self.img)
        if self.type_detect.currentText() == 'Детектирование синтеза речи':
            self.probability_bonafide.setText('Вероятность естественной речи составляет - '
                                              + ' ' + str(round(bonafide * 100, 2)) + '%')
            self.probability_spoof.setText('Вероятность синтезированной речи составляет - '
                                           + ' ' + str(round(spoof * 100, 2)) + '%')
        elif self.type_detect.currentText() == 'Детектирование модификации речи':
            self.probability_bonafide.setText('Вероятность естественной речи составляет - '
                                              + ' ' + str(round(bonafide * 100, 2)) + '%')
            self.probability_spoof.setText('Вероятность модификации речи составляет - '
                                           + ' ' + str(round(spoof * 100, 2)) + '%')
        self.probability_bonafide.setVisible(True)
        self.probability_spoof.setVisible(True)
        self.heatmap.setVisible(True)
        self.heatmap_filters.setVisible(True)
        self.label_analisys.setVisible(False)

        if self.type_detect.currentText() == 'Детектирование синтеза речи':
            with open(os.path.join(os.path.abspath(os.curdir), 'App', 'tresholds'), 'r') as text_file:
                tresholds_ = text_file.read()
        elif self.type_detect.currentText() == 'Детектирование модификации речи':
            with open(os.path.join(os.path.abspath(os.curdir), 'App', 'tresholds_mod'), 'r') as text_file:
                tresholds_ = text_file.read()
        if bonafide >= float(tresholds_):
            self.label_ok.setVisible(True)
            self.label_result.setStyleSheet('color: Green')
            self.label_result.setText('     Естественная речь')
            self.label_result.setVisible(True)
        else:
            self.label_stop.setVisible(True)
            self.label_result.setStyleSheet('color: Red')
            if self.type_detect.currentText() == 'Детектирование синтеза речи':
                self.label_result.setText('Синтезированная речь')
            elif self.type_detect.currentText() == 'Детектирование модификации речи':
                self.label_result.setText('Модифицированная речь')
            self.label_result.setVisible(True)

        self.set_dis(False)
        
    def open_file(self):
        if os.path.exists(os.path.join('App', 'temp', 'filters')):
            shutil.rmtree(os.path.join('App', 'temp', 'filters'))
        list_file = os.listdir(os.path.join('App', 'temp'))
        for i in list_file:
            if i != 'directory.txt':
                os.remove(os.path.join('App', 'temp', i))
        self.label_result.setVisible(False)
        self.probability_bonafide.setVisible(False)
        self.probability_spoof.setVisible(False)
        self.label_ps.setVisible(False)
        self.product_spectrum_img.setVisible(False)
        self.label_ps_crop.setVisible(False)
        self.product_spectrum_img_crop.setVisible(False)
        self.analisys_file_btn.setVisible(False)
        self.play_audio.setVisible(False)
        self.stop_audio.setVisible(False)
        self.open_phase_spectrs.setVisible(False)
        self.open_setting.setVisible(False)
        self.label_ok.setVisible(False)
        self.label_stop.setVisible(False)
        self.heatmap.setVisible(False)
        self.heatmap_filters.setVisible(False)
        self.type_detect.setVisible(False)

        filter_f = 'Audio file (*.wav *.WAV *.Wav *.wAv *.WaV *.wAV *.WAv)'
        options = QFileDialog.DontUseNativeDialog
        self.filename_and_dir, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Открыть  аудиофайл", self.directory,
                                                                         filter_f, options=options)
        self.line_filename.setText(self.filename_and_dir)
        if (self.filename_and_dir != u'') and (self.filename_and_dir != u'Аудиофайл не выбран'):
            self.filename_audio = self.filename_and_dir.split("/")[-1]
            self.directory = self.filename_and_dir.split(self.filename_audio)[0]
            with open(os.path.join(os.path.abspath(os.curdir), 'App', 'temp', 'directory.txt'), 'w') as text_file:
                text_file.writelines(self.directory)
            try:
                self.name_dir_after_preprocess, self.name_file_after_preprocess \
                    = preprocess(self.directory, self.filename_audio)
                read_and_transform_ps(self.name_dir_after_preprocess, self.name_file_after_preprocess)

                self.img = open_img(self.name_dir_after_preprocess, self.name_file_after_preprocess)
                self.img.save(os.path.join(self.name_dir_after_preprocess,
                                           self.name_file_after_preprocess.split('.wav')[0] + '_crop.png'))

                pixmap = QPixmap(os.path.join(self.name_dir_after_preprocess,
                                              self.name_file_after_preprocess.split('.wav')[0] + '.png'))
                self.product_spectrum_img.setPixmap(pixmap)
                self.label_ps.setVisible(True)
                self.product_spectrum_img.setVisible(True)

                pixmap_crop = QPixmap(os.path.join(self.name_dir_after_preprocess,
                                                   self.name_file_after_preprocess.split('.wav')[0] + '_crop.png'))
                self.product_spectrum_img_crop.setPixmap(pixmap_crop)
                self.label_ps_crop.setVisible(True)
                self.product_spectrum_img_crop.setVisible(True)

                self.img = tf.convert_to_tensor(self.img, dtype=tf.float32)
                self.img = tf.expand_dims(self.img, 0)

                self.play_audio.setVisible(True)
                self.stop_audio.setVisible(True)
                self.open_phase_spectrs.setVisible(True)
                self.open_setting.setVisible(True)
                self.label_img.setVisible(False)
                self.analisys_file_btn.setVisible(True)
                self.type_detect.setVisible(True)

            except Exception as e:
                message_info(str(e))

    def closeEvent(self, event):
        if os.path.exists(os.path.join('App', 'temp', 'filters')):
            shutil.rmtree(os.path.join('App', 'temp', 'filters'))
        list_file = os.listdir(os.path.join('App', 'temp'))
        for i in list_file:
            if i != 'directory.txt':
                os.remove(os.path.join('App', 'temp', i))
        event.accept()


class PhaseSpectrum(QtWidgets.QWidget):
    def __init__(self, name_audiofile):
        super(PhaseSpectrum, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.curdir), 'App', 'form', 'win_spectr.ui'), self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowTitle(name_audiofile)
        self.name_audiofile = name_audiofile

        self.show_bpd.clicked.connect(self.show_bpd_func)
        self.show_gd.clicked.connect(self.show_gd_func)
        self.show_lms.clicked.connect(self.show_lms_func)
        self.show_psp.clicked.connect(self.show_psp_func)
        self.show_mgd.clicked.connect(self.show_mgd_func)
        self.show_if.clicked.connect(self.show_if_func)
        self.show_cepstr.clicked.connect(self.show_cepstr_func)

        try:
            pixmap_bpd = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_bpd.png'))
            self.label_bpd.setPixmap(pixmap_bpd)
        except Exception as e:
            message_info(str(e))

        try:
            pixmap_mgd = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_mgd.png'))
            self.label_mgd.setPixmap(pixmap_mgd)
        except Exception as e:
            message_info(str(e))

        try:
            pixmap_if = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_if.png'))
            self.label_if.setPixmap(pixmap_if)
        except Exception as e:
            message_info(str(e))

        try:
            pixmap_gd = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_gd.png'))
            self.label_gd.setPixmap(pixmap_gd)
        except Exception as e:
            message_info(str(e))

        try:
            pixmap_psp = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_psp.png'))
            self.label_psp.setPixmap(pixmap_psp)
        except Exception as e:
            message_info(str(e))

        try:
            pixmap_lms = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_lms.png'))
            self.label_lms.setPixmap(pixmap_lms)
        except Exception as e:
            message_info(str(e))
        try:
            pixmap_cepstr = QPixmap(os.path.join('App', 'temp', name_audiofile.split('.wav')[0] + '_cepstr.png'))
            self.label_cepstr.setPixmap(pixmap_cepstr)
        except Exception as e:
            message_info(str(e))

    def show_bpd_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_bpd.pickle'), 'rb') as f:
            bpd_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(bpd_, cmap='jet', origin='lower')
        plt.show()

    def show_gd_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_gd.pickle'), 'rb') as f:
            gd_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(gd_, cmap='jet', origin='lower')
        plt.show()

    def show_lms_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_lms.pickle'), 'rb') as f:
            lms_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(lms_, cmap='jet', origin='lower')
        plt.show()

    def show_psp_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_psp.pickle'), 'rb') as f:
            psp_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(psp_, cmap='jet', origin='lower')
        plt.show()

    def show_mgd_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_mgd.pickle'), 'rb') as f:
            mgd_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(mgd_, cmap='jet', origin='lower')
        plt.show()

    def show_if_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_if.pickle'), 'rb') as f:
            if_ = pickle.load(f)
        plt.figure()
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(if_, cmap='jet', origin='lower')
        plt.show()

    def show_cepstr_func(self):
        with open(os.path.join('App', 'temp', self.name_audiofile.split('.wav')[0] + '_cepstr.pickle'), 'rb') as f:
            cepstr_ = pickle.load(f)
        freq_value = np.zeros(cepstr_.shape[0] + 1)
        for i in np.arange(cepstr_.shape[0] + 1):
            freq_value[i] = round(16000 / (16000 / 70 - i))
        plt.figure()
        ax = plt.gca()
        ax.set_yticks(np.arange(freq_value.shape[0]))
        ax.set_yticklabels(freq_value)
        plt.gcf().canvas.set_window_title(self.name_audiofile)
        plt.imshow(cepstr_, cmap='jet', origin='lower')
        plt.show()


class SettingMainTone(QtWidgets.QWidget):
    def __init__(self, name_audiofile):
        super(SettingMainTone, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.curdir), 'App', 'form', 'set.ui'), self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.name_audiofile = name_audiofile

        self.ok.clicked.connect(self.ok_func)
        self.height_peak.returnPressed.connect(self.ok_func)
        self.dist_peak.returnPressed.connect(self.ok_func)

    def ok_func(self):
        try:
            treshhold = int(self.height_peak.text())
            try:
                dist = int(self.dist_peak.text())
                with open(os.path.join('App', 'temp', 'set_main_tone.txt'), 'w') as text_file:
                    text_file.writelines(self.height_peak.text() + '\n' + self.dist_peak.text())
                rate, audio = wav.read(os.path.join('App', 'temp', self.name_audiofile))
                metki, xz = sig.find_peaks(audio, height=treshhold, distance=dist)

                plt.figure()
                plt.gcf().canvas.set_window_title(self.name_audiofile)
                plt.plot(metki, audio[metki], "o")
                plt.plot(audio)
                plt.show()

            except ValueError:
                message_info('Введите число в поле /Расстояния между пиками/')
        except ValueError:
            message_info('Введите число в поле /Порог амплитуды пиков/')


class HeatMapsFilters(QtWidgets.QWidget):
    def __init__(self, sort_list_filters):
        super(HeatMapsFilters, self).__init__()
        uic.loadUi(os.path.join(os.path.abspath(os.curdir), 'App', 'form', 'heatmap_filters.ui'), self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowTitle('Тепловые карты акивации фильтров')
        self.sort_list_filters = sort_list_filters
        filters_ = [str(i) for i in range(1, len(self.sort_list_filters) + 1)]
        self.number_of_filter.addItems(filters_)
        if os.path.isfile(os.path.join('App', 'temp', 'filters', self.sort_list_filters[0])):
            pixmap_heat = QPixmap(QPixmap(os.path.join('App', 'temp', 'filters', self.sort_list_filters[0])))
            self.filts.setPixmap(pixmap_heat)

        else:
            message_info('Не найден файл: ' + str(os.path.join('App', 'temp', 'filters', sort_list_filters[0])))

        self.number_of_filter.currentIndexChanged.connect(self.change_combobox)
        self.show_filter.clicked.connect(self. show_filter_func)

    def show_filter_func(self):
        if os.path.isfile(os.path.join('App', 'temp', 'filters',
                                       self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])):
            filter_ = cv2.imread(os.path.join('App', 'temp', 'filters',
                                              self.sort_list_filters[int(self.number_of_filter.currentText()) - 1]))
            plt.figure()
            ax = plt.gca()
            plt.gcf().canvas.set_window_title('Карта активации фильтра № ' +
                                              str(int(self.number_of_filter.currentText())))
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(plt.imread(os.path.join('App', 'temp', 'filters',
                                               self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])))
            plt.show()
        else:
            message_info('Не найден файл: ' +
                         str(os.path.join('App', 'temp', 'filters',
                                          self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])))

    def change_combobox(self):
        if os.path.isfile(os.path.join('App', 'temp', 'filters',
                                       self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])):
            pixmap_heat = QPixmap(QPixmap(
                os.path.join('App', 'temp', 'filters',
                             self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])))
            self.filts.setPixmap(pixmap_heat)
        else:
            message_info('Не найден файл: ' +
                         str(os.path.join('App', 'temp', 'filters',
                                          self.sort_list_filters[int(self.number_of_filter.currentText()) - 1])))
        # pic.move(20, 224)

        # self.ok.clicked.connect(self.ok_func)
        # self.height_peak.returnPressed.connect(self.ok_func)
        # self.dist_peak.returnPressed.connect(self.ok_func)

    # def ok_func(self):
        # try:
        #     treshhold = int(self.height_peak.text())
        #     try:
        #         dist = int(self.dist_peak.text())
        #         with open(os.path.join('App', 'temp', 'set_main_tone.txt'), 'w') as text_file:
        #             text_file.writelines(self.height_peak.text() + '\n' + self.dist_peak.text())
        #         rate, audio = wav.read(os.path.join('App', 'temp', self.name_audiofile))
        #         metki, xz = sig.find_peaks(audio, height=treshhold, distance=dist)
        #
        #         plt.figure()
        #         plt.gcf().canvas.set_window_title(self.name_audiofile)
        #         plt.plot(metki, audio[metki], "o")
        #         plt.plot(audio)
        #         plt.show()
        #
        #     except ValueError:
        #         message_info('Введите число в поле /Расстояния между пиками/')
        # except ValueError:
        #     message_info('Введите число в поле /Порог амплитуды пиков/')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Auth = FileAnalisys()
    Auth.show()
    sys.exit(app.exec_())
