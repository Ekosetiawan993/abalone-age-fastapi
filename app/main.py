from joblib import load
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/mlp_abalone_age_prediction-{__version__}.sav", "rb") as f:
    model = load(f)


def mlp_predict(abalone_features):
    """
    abalone_features : List of seven abalone's features
    """
    predicted_age = model.predict([abalone_features])

    if predicted_age[0] <= 0:
        return f"There is an issue with the input data."

    return predicted_age[0]


app = FastAPI()

# Mount the "static" folder to serve CSS and other static files
app.mount(
    "/static", StaticFiles(directory=f"{BASE_DIR}/static"), name="static")

# Mount the "templates" folder to load HTML templates
templates = Jinja2Templates(directory=f"{BASE_DIR}/templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, })


@app.post("/predict", response_class=HTMLResponse, )
def predict(request: Request, length: float = Form(...), diameter: float = Form(...), height: float = Form(...), whole_weight: float = Form(...), shucked_weight: float = Form(...), viscera_weight: float = Form(...), shell_weight: float = Form(...)):  # sentence: str = Form(...)

    abalone_features = [length, diameter, height, whole_weight,
                        shucked_weight, viscera_weight, shell_weight]
    prediction = mlp_predict(abalone_features)
    prediction = round(prediction, 2)

    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "prediction": prediction},
    )
