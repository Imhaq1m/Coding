# -----------------------------
# 1. Import Libraries
# -----------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 2. Define Paths and Parameters
# -----------------------------
train_dir = 'dataset/train'
test_dir = 'dataset/test'

img_size = 48   # VGG16 can work with smaller images too!
batch_size = 64
epochs = 25
num_classes = 7  # 7 emotions

# -----------------------------
# 3. Data Augmentation with VGG Preprocessing
# -----------------------------
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # VGG-specific preprocessing
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),     # Resize images
    color_mode='rgb',                     # VGG uses RGB images
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = datagen_test.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# 4. Load VGG16 Base Model
# -----------------------------
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

# Freeze base layers for first stage of training
base_model.trainable = False

# -----------------------------
# 5. Add Custom Layers on Top
# -----------------------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)   # You can also try 512 units
x = Dropout(0.5)(x)                    # Helps prevent overfitting
output = Dense(num_classes, activation='softmax')(x)

model = Model(base_model.input, output)

# -----------------------------
# 6. Compile the Model
# -----------------------------
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -----------------------------
# 7. Train the Model
# -----------------------------
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator
)

# -----------------------------
# 8. Evaluate Final Accuracy
# -----------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
