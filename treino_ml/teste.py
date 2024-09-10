import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Configurações
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 1  # Número de classes a serem treinadas

# Preparando os dados (ajuste para suas pastas)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('C:\\Users\\jhennifer.nascimento\\nfs\\treino_ml\\train\\sp', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory('C:\\Users\\jhennifer.nascimento\\nfs\\treino_ml\\validation\\sp', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Carregar ResNet50 sem a cabeça (camadas densas finais)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas no topo
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(train_generator, validation_data=val_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, 
          validation_steps=val_generator.samples // BATCH_SIZE, epochs=EPOCHS)

# Salvar o modelo treinado
model.save('meu_modelo_avancado.h5')
