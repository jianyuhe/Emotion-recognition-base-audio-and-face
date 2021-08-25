"""
Description: 训练人脸表情识别程序
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import load_affectnet
from load_and_process import preprocess_input
from cnn import mini_XCEPTION
from cnn import big_XCEPTION
from cnn import ms_model
import cnn
from sklearn.model_selection import train_test_split
#from keras.utils import multi_gpu_model


# 参数
batch_size = 320
num_epochs = 10000
input_shape = (48, 48, 1) #(224, 224, 3)
validation_split = .2
verbose = 1
num_classes = 7 #8
patience = 50
base_path = 'models/'


# 构建模型.
model = ms_model(num_classes, input_shape)
# model = mini_XCEPTION(input_shape, num_classes)
#model = cnn.simple_CNN(input_shape, num_classes)
#model = cnn.simpler_CNN(input_shape, num_classes)
#model = cnn.tiny_XCEPTION(input_shape, num_classes)
#model = multi_gpu_model(mini_model, gpus=1)
model.compile(optimizer='adam', # 优化器采用adam
              loss='categorical_crossentropy', # 多分类的对数损失函数
              metrics=['accuracy'])
model.summary()




# 定义回调函数 Callbacks 用于训练过程
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4),
                              verbose=1)
# 模型位置及命名
trained_models_path = base_path + 'e230_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                    save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]



# 载入数据集
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# 划分训练、测试集
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

# 载入affectnet数据集
# xtrain, ytrain = load_affectnet('train_set',8)
# xtest, ytest = load_affectnet('val_set',8)
# xtrain = preprocess_input(xtrain)
# xtest = preprocess_input(xtest)


# 图片产生器，在批量中对数据进行增强，扩充数据集大小
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# 利用数据增强进行训练
#model = multi_gpu_model(model, 1)
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))



