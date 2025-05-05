import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json

# 数据集路径
data_dir = 'archive/images'
meta_dir = 'archive/meta/meta'

# 读取类别信息
with open(os.path.join(meta_dir, 'classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = len(classes)
print(f"共有{num_classes}个食品类别")

# 图像参数
img_height, img_width = 224, 224
batch_size = 32

# 数据增强和预处理
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 训练集
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 验证集
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 构建MobileNetV2模型（迁移学习）
base_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# 冻结预训练层
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义分类层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 最终模型
model = Model(inputs=base_model.input, outputs=predictions)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
callbacks = [
    ModelCheckpoint('food_mobilenet_model.h5', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# 训练模型
epochs = 100  # 设置较大的值，但依赖早停机制
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1  # 显示进度条
)

# 微调：解冻部分层进行进一步训练
print("开始微调模型...")
# 解冻MobileNetV2的最后几层
for layer in model.layers[-20:]:
    layer.trainable = True

# 使用较小的学习率重新编译
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # 更小的学习率
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 继续训练
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # 微调阶段轮数也可以设大些，依赖早停
    callbacks=callbacks,
    verbose=1
)

# 保存最终模型
model.save('food101_mobilenet_final.h5')
print("模型训练完成并已保存!")