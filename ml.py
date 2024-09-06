import streamlit as st
import tensorflow as tf
import numpy as np
import keras 


img_file_buffer = st.camera_input("tirar foto")


if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3)

    img_resized = tf.image.resize(img_tensor, [224, 224])

    img_preprocessado = preprocess_input(tf.expand_dims(img_resized, axis=0))

    model = MobileNetV2(weights= 'imagenet')

    previsoes = model.predict(img_preprocessado)

    decoded_previsoes= decode_predictions(previsoes, top=1)[0]
    objeto_dominante = decoded_previsoes[0][1]
    objeto_score = decoded_previsoes[0][2]

    st.write(f"objeto dominante: {objeto_dominante} ({objeto_score * 100:.2f}%)")

