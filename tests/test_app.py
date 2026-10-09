from app import app


def test_home_page():
    response = app.test_client().get("/")
    assert response.status_code == 200


def test_prediction_page():
    response = app.test_client().get("/predictdata")
    assert response.status_code == 200