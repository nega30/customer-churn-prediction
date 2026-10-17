\# Customer Churn Prediction



A machine learning project that predicts whether a telecom customer is likely to leave a service provider using customer demographics, service subscriptions, contract details, tenure, and billing information.



\## Project Overview



Customer churn is an important business problem for telecom companies. This project explores customer churn patterns, trains a Random Forest classifier, and exposes predictions through a REST API built with FastAPI.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Data analysis:\*\* Pandas, NumPy

\- \*\*Visualization:\*\* Matplotlib, Seaborn

\- \*\*Machine learning:\*\* Scikit-learn

\- \*\*Model persistence:\*\* Joblib

\- \*\*API:\*\* FastAPI, Uvicorn

\- \*\*Testing:\*\* Pytest



\## Dataset



\- Source: IBM Telco Customer Churn dataset

\- Records: 7,043 customers

\- Features: 20 input columns, excluding the customer ID

\- Target: `Churn` (`Yes` or `No`)



Data cleaning included converting `TotalCharges` to numeric values and handling 11 blank values associated with zero-tenure customers by assigning zero accumulated charges.



\## Exploratory Data Analysis



Key findings from the dataset:



\- Overall churn rate: \*\*26.54%\*\*

\- Month-to-month contract churn rate: \*\*42.7%\*\*

\- One-year contract churn rate: \*\*11.3%\*\*

\- Two-year contract churn rate: \*\*2.8%\*\*

\- Average tenure of churned customers: \*\*17.98 months\*\*

\- Average tenure of retained customers: \*\*37.57 months\*\*



These are observed associations in the dataset and do not establish causation.



\## Model and Evaluation



The project uses a Random Forest classifier with a stratified 80/20 train-test split.



| Metric | Result |

|---|---:|

| Accuracy | 77.00% |

| Churn precision | 56% |

| Churn recall | 63% |

| Churn F1-score | 0.59 |

| ROC-AUC | 0.8225 |



The current model is a baseline. In particular, its churn recall indicates that some customers who actually churn are missed.



\## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/` | API welcome message |

| GET | `/health` | Health check |

| POST | `/predict` | Predict customer churn |



Interactive API documentation is available at `/docs` while the local server is running.



\## Run Locally



\### 1. Clone the repository



```bash

git clone https://github.com/nega30/customer-churn-prediction.git

cd customer-churn-prediction

```



\### 2. Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

python -m pip install -r requirements.txt

```



\### 4. Start the API



```bash

python -m uvicorn src.app:app --reload

```



Open `http://127.0.0.1:8000/docs` to test predictions.



\## Run Tests



```bash

python -m pytest src/test\_app.py -v

```



The current test suite contains three tests covering the home endpoint, health endpoint, and prediction endpoint.



\## Project Structure



```text

customer-churn-prediction/

├── data/

│   ├── raw/

│   └── processed/

├── models/

│   └── churn\_model.joblib

├── reports/

│   └── figures/

├── src/

│   ├── inspect\_data.py

│   ├── clean\_data.py

│   ├── eda.py

│   ├── train\_model.py

│   ├── app.py

│   └── test\_app.py

├── .gitignore

├── requirements.txt

└── README.md

```



\## Limitations and Future Improvements



\- Compare Random Forest with Logistic Regression and other baseline models.

\- Tune the classification threshold based on business costs.

\- Improve churn recall and evaluate model calibration.

\- Add input validation, model monitoring, and more automated tests.

\- Build a frontend dashboard for interactive predictions.

\- Add reproducible data and model training instructions.



\## Disclaimer



This project is for educational and portfolio purposes. Predictions are estimates based on historical data and should not be treated as guarantees of future customer behavior.



