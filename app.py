from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from credit_risk.constants import APP_HOST, APP_PORT
from credit_risk.pipline.prediction_pipeline import CreditRiskData, CreditRiskClassifier
from credit_risk.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request

        self.limit_bal: Optional[str] = None
        self.sex: Optional[str] = None
        self.education: Optional[str] = None
        self.marriage: Optional[str] = None
        self.age: Optional[str] = None

        self.payment_status_sep: Optional[str] = None
        self.payment_status_aug: Optional[str] = None
        self.payment_status_jul: Optional[str] = None
        self.payment_status_jun: Optional[str] = None
        self.payment_status_may: Optional[str] = None
        self.payment_status_apr: Optional[str] = None

        self.bill_statement_sep: Optional[str] = None
        self.bill_statement_aug: Optional[str] = None
        self.bill_statement_jul: Optional[str] = None
        self.bill_statement_jun: Optional[str] = None
        self.bill_statement_may: Optional[str] = None
        self.bill_statement_apr: Optional[str] = None

        self.previous_payment_sep: Optional[str] = None
        self.previous_payment_aug: Optional[str] = None
        self.previous_payment_jul: Optional[str] = None
        self.previous_payment_jun: Optional[str] = None
        self.previous_payment_may: Optional[str] = None
        self.previous_payment_apr: Optional[str] = None

    async def get_credit_risk_data(self):
        form = await self.request.form()

        self.limit_bal = form.get("limit_bal")
        self.sex = form.get("sex")
        self.education = form.get("education")
        self.marriage = form.get("marriage")
        self.age = form.get("age")

        self.payment_status_sep = form.get("payment_status_sep")
        self.payment_status_aug = form.get("payment_status_aug")
        self.payment_status_jul = form.get("payment_status_jul")
        self.payment_status_jun = form.get("payment_status_jun")
        self.payment_status_may = form.get("payment_status_may")
        self.payment_status_apr = form.get("payment_status_apr")

        self.bill_statement_sep = form.get("bill_statement_sep")
        self.bill_statement_aug = form.get("bill_statement_aug")
        self.bill_statement_jul = form.get("bill_statement_jul")
        self.bill_statement_jun = form.get("bill_statement_jun")
        self.bill_statement_may = form.get("bill_statement_may")
        self.bill_statement_apr = form.get("bill_statement_apr")

        self.previous_payment_sep = form.get("previous_payment_sep")
        self.previous_payment_aug = form.get("previous_payment_aug")
        self.previous_payment_jul = form.get("previous_payment_jul")
        self.previous_payment_jun = form.get("previous_payment_jun")
        self.previous_payment_may = form.get("previous_payment_may")
        self.previous_payment_apr = form.get("previous_payment_apr")


@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "creditrisk.html",
        {"request": request, "context": "Rendering"}
    )


@app.get("/train")
async def train_route_client():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predict_route_client(request: Request):
    try:
        form = DataForm(request)
        await form.get_credit_risk_data()

        credit_risk_data = CreditRiskData(
            limit_bal=float(form.limit_bal),
            sex=form.sex,
            education=form.education,
            marriage=form.marriage,
            age=float(form.age),

            payment_status_sep=form.payment_status_sep,
            payment_status_aug=form.payment_status_aug,
            payment_status_jul=form.payment_status_jul,
            payment_status_jun=form.payment_status_jun,
            payment_status_may=form.payment_status_may,
            payment_status_apr=form.payment_status_apr,

            bill_statement_sep=float(form.bill_statement_sep),
            bill_statement_aug=float(form.bill_statement_aug),
            bill_statement_jul=float(form.bill_statement_jul),
            bill_statement_jun=float(form.bill_statement_jun),
            bill_statement_may=float(form.bill_statement_may),
            bill_statement_apr=float(form.bill_statement_apr),

            previous_payment_sep=float(form.previous_payment_sep),
            previous_payment_aug=float(form.previous_payment_aug),
            previous_payment_jul=float(form.previous_payment_jul),
            previous_payment_jun=float(form.previous_payment_jun),
            previous_payment_may=float(form.previous_payment_may),
            previous_payment_apr=float(form.previous_payment_apr),
        )

        credit_risk_df = credit_risk_data.get_credit_risk_input_data_frame()

        model_predictor = CreditRiskClassifier()
        value = model_predictor.predict(dataframe=credit_risk_df)[0]

        status = None
        if value == 1:
            status = "Default Risk Detected"
        else:
            status = "No Default Risk Detected"

        return templates.TemplateResponse(
            "creditrisk.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)