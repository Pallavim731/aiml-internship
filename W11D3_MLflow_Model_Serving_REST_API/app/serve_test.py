import requests

url = "http://127.0.0.1:5001/invocations"

data = {
    "dataframe_split": {
        "columns": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ],
        "data": [
            [5.1, 3.5, 1.4, 0.2]
        ]
    }
}

response = requests.post(
    url,
    json=data
)

print("Status code:", response.status_code)
print("Response:", response.text)