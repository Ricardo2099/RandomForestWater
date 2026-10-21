import requests

url = "http://127.0.0.1:8000/score"
datos = {
    "ph": 7.14,
    "hardness": 440.98,
    "tds": 614.26,
    "chlorine": 2.12,
    "sulfate": 208.39,
    "conductivity": 350.92,
    "organic_carbon": 7.22,
    "trihalomethanes": 17.32,
    "turbidity": 2.23,
}

respuesta = requests.post(url, json=datos, timeout=10)
respuesta.raise_for_status()
print(respuesta.json())
