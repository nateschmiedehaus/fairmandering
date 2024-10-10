import pytest
from fairmandering import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Fairmandering' in response.data

def test_run_redistricting(client):
    response = client.post('/run', data={'state_fips': '06'})
    assert response.status_code == 200 or response.status_code == 302
