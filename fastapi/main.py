from fastapi import FastAPI
import requests

app = FastAPI()

@app.post("/trigger")
def trigger_dag():
    # 觸發 Airflow DAG (用 REST API)
    url = "http://airflow:8080/api/v1/dags/hello_world/dagRuns"
    r = requests.post(
        url,
        auth=("admin", "admin"),
        json={"conf": {}}
    )
    return {"status": r.status_code, "data": r.json()}
