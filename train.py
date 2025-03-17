import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri yolu
data_dir = 'flowers'

# Veri artırma ve normalleştirme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Aşırı dönüşüm değerlerini azalttım
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Veri yükleyicileri
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Sınıf sayısını al
num_classes = len(train_generator.class_indices)

# Transfer öğrenme için önceden eğitilmiş model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Önceden eğitilmiş katmanları dondur

# Daha basit model yapısı
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),  # Normalizasyon ekleyin
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Gradyan kırpma için özel optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,  # Daha düşük öğrenme oranı
    clipnorm=1.0  # Gradyan kırpma ekleyin
)

# Modeli derleme
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Öğrenme oranını azaltan callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,  # Daha erken müdahale
    min_lr=0.00001,
    verbose=1  # Öğrenme oranı değişimlerini göster
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint - en iyi modeli kaydet
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_flower_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Model eğitimi
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

# Modeli kaydet
model.save('flower_classification_model.keras')  # .keras formatında kaydet