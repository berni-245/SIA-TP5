# TP5 SIA - Deep Learning

## 👋 Introducción

Trabajo práctica para la materia de Sistemas de Inteligencia Artificial en el ITBA. Se buscó implementar dos tipos de autoencoders, el autoencoder tradicional y el autoencoder variacional. En los mains se utiliza de dataset unas fonts impresas en matrcies 7x5. Más información del formato en este [lab](fonts.ipynb)

Este fue el [Enunciado](docs/SIATP5.pdf)

### ❗ Requisitos

- Python3 (La aplicación se probó en la versión de Python 3.11.*)
- pip3
- [pipenv](https://pypi.org/project/pipenv)

### 💻 Instalación

En caso de no tener python, descargarlo desde la [página oficial](https://www.python.org/downloads/release/python-3119/)

Utilizando pip (o pip3 en mac/linux) instalar la dependencia de **pipenv**:

```sh
pip install pipenv
```

Parado en la carpeta del proyecto ejecutar:

```sh
pipenv install
```

para instalar las dependencias necesarias en el ambiente virtual.

## 🛠️ Configuración

Todos los mains tienen su correspondiente config file con el mismo sufijo. Todos los configs comparten estos hiperparámetros modificables:
- `hidden_layers`: las capas ocultas entre el inicio del encoder y el espacio latente (sin contar estos), luego se verá espejado al decoder
- `latent_dim`: la dimensión del espacio latente
- `max_epochs`: la máxima cantidad de épocas para cortar
- `learning_rate`: el learn rate
- `min_error`: el mínimo error para cortar

Para el caso del config_DAE se tiene 3 elementos configurables adicionales:

- `min_train_noise_level`: el mínimo noise que se le aplicarán a los datos de entrenamientos
- `max_train_noise_level`: el máximo noise que se le aplicarán a los datos de entrenamientos
- `noise_level_for_test`: el noise que se le aplicará al elemento en el índice elegido, para comparar si se reconstruye o no

Finalmente para el caso del config_VAE se tiene un elemento configurable:
- `batch_size`: esto es solo para este caso por tema optimización pero se podrá elegir un batch_size para actualizar los gradientes tras la cantidad de datos especificada

## 🏃 Ejecución

Para probar la aplicación, correr:
```shell
pipenv run python <main deseado>
```

Con <main deseado\> uno de los siguientes:
- `main_AE`: main de autoencoder básico
- `main_DAE`: main de autoencoder con denoising
- `main_VAE`: main de autoencoder variacional