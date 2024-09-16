import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Carregar o modelo treinado no Teachable Machine
model = load_model('path_to_your_model.h5')

# Função para pré-processar a imagem para o modelo
def preprocess_image(image_path):
    # Carregar a imagem e ajustar o tamanho para o esperado pelo modelo
    image = Image.open(image_path)
    size = (224, 224)  # Tamanho padrão do modelo
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    # Converter a imagem em um array de NumPy e normalizar os valores
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão extra
    img_array = img_array.astype(np.float32) / 255.0  # Normalizar os valores para [0,1]
    
    return img_array

# Função para fazer a predição
def predict_image(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction

# Exemplo de uso
image_path = 'path_to_your_image.jpg'  # Caminho para sua imagem
result = predict_image(image_path)

# Exibir o resultado
print(f'Resultado da predição: {result}')
