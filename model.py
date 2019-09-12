import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import numpy as np
import os
import split_folders
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

# function for creating the spectrograms
def create_spectrograms(folder,spectrogram_path,audio_path):
    print('Processing folder '+ folder)
    os.mkdir(spectrogram_path+folder)
    for audio_file in os.listdir(audio_path+folder):
        samples, sample_rate = librosa.load(audio_path+folder+'/'+audio_file)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename  = spectrogram_path+folder+'/'+audio_file.replace('.wav','.png')
        S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

# creating spectrograms for train set and saving as png images
folders = os.listdir('share/train/')
for folder in folders:
    create_spectrograms(folder,spectrogram_path = 'images/train/',audio_path = 'share/train/')

# creating spectrograms for test set and saving as png images
for folder in os.listdir('share/test/'):
    create_spectrograms(folder,spectrogram_path = 'images/test/' ,audio_path = 'share/test/')

# splitting the training images into train and validation folders
split_folders.ratio('images/train', output="output", seed=1337, ratio=(.7, .3))


# creating an image classification model using fastai library and resnet34 model for transfer learning
path = Path('output/')
bs = 16 # batch size
data = ImageDataBunch.from_folder(path, size=224, bs=bs)
learn = create_cnn(data, models.resnet34, metrics=[error_rate,accuracy])
learn.fit_one_cycle(20)
# unfreezing the layers for the weights to move around a bit
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
# fitting again on specific learning rate
learn.fit_one_cycle(8, max_lr=slice(1e-6,1e-4))
learn.fit_one_cycle(8)
# saving the model
learn.save('model-res34_9666')

# prediction on the test images
test_files = os.listdir('images/test/test/')
preds = []
for file in test_files:
    img = open_image(f'images/test/test/'+file)
    p = learn.predict(img)
    preds.append(str(p[0]))

sub = pd.DataFrame()
sub['File_name'] = test_files
sub['labels'] = preds
sub['File_name'] = sub['File_name'].apply(lambda x: x.replace('.png','.wav'))
sub.to_csv('upload/Mayank_Siddharth.csv', index = False)
