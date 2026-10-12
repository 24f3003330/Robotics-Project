import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app import app, robot, HARDWARE_ENABLED, robot_poses, EMOTION_TO_POSE

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_ble():
    with patch("app.BleakClient") as MockClient, \
         patch("app.BleakScanner.discover", new_callable=AsyncMock) as MockScanner:
        
        # Setup mock client instance
        mock_instance = MockClient.return_value
        mock_instance.is_connected = True
        mock_instance.connect = AsyncMock()
        mock_instance.disconnect = AsyncMock()
        mock_instance.write_gatt_char = AsyncMock()
        
        # Setup mock characteristics
        mock_char = MagicMock()
        mock_char.uuid = "0000ffe1-0000-1000-8000-00805f9b34fb"
        mock_char.handle = 42
        mock_char.properties = ["write", "write-without-response"]
        
        mock_service = MagicMock()
        mock_service.uuid = "0000ffe0-0000-1000-8000-00805f9b34fb"
        mock_service.characteristics = [mock_char]
        mock_instance.services = [mock_service]
        
        yield MockClient, MockScanner

def test_robot_hardware_enable(client):
    response = client.post("/robot/hardware_enable", json={"enable": True})
    assert response.status_code == 200
    assert response.json["hardware_enabled"] is True

    response = client.post("/robot/hardware_enable", json={"enable": False})
    assert response.status_code == 200
    assert response.json["hardware_enabled"] is False
    robot.connected = False
    robot.busy = False

def test_robot_dry_run_move(client):
    client.post("/robot/hardware_enable", json={"enable": False})
    response = client.post("/robot/move", json={"positions": {"1": 500, "2": 500}})
    assert response.status_code == 200
    assert "DRY RUN" in response.json["message"]

def test_robot_move_validation(client):
    client.post("/robot/hardware_enable", json={"enable": True})
    
    # Test invalid servo ID
    response = client.post("/robot/move", json={"positions": {"7": 500}})
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert "Invalid servo ID" in response.json["error"]

    # Test invalid position
    response = client.post("/robot/move", json={"positions": {"1": 1500}})
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert "Invalid position" in response.json["error"]
    
    # Negative position
    response = client.post("/robot/move", json={"positions": {"1": -50}})
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert "Invalid position" in response.json["error"]
    
    client.post("/robot/hardware_enable", json={"enable": False})

def test_robot_dry_run_home(client):
    client.post("/robot/hardware_enable", json={"enable": False})
    response = client.get("/robot/home")
    assert response.status_code == 200
    assert "DRY RUN" in response.json["message"]

def test_robot_dry_run_emotion(client):
    client.post("/robot/hardware_enable", json={"enable": False})
    response = client.get("/robot/run/happy")
    assert response.status_code == 200
    assert "DRY RUN" in response.json["message"]

def test_robot_connect_success(client, mock_ble):
    MockClient, MockScanner = mock_ble
    
    response = client.post("/robot/connect", json={"address": "00:11:22:33:44:55"})
    assert response.status_code == 200
    if not response.json["ok"]:
        print(response.json)
    assert response.json["ok"] is True
    assert robot.connected is True
    assert robot.write_handle == 42
    
    # Cleanup
    client.get("/robot/disconnect")

def test_robot_connect_failure(client, mock_ble):
    MockClient, MockScanner = mock_ble
    mock_instance = MockClient.return_value
    mock_instance.is_connected = False # Simulate failure to connect
    
    response = client.post("/robot/connect", json={"address": "00:11:22:33:44:55"})
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert robot.connected is False

def test_robot_move_disconnected(client):
    client.post("/robot/hardware_enable", json={"enable": True})
    robot.connected = False
    
    response = client.post("/robot/move", json={"positions": {"1": 500}})
    assert response.status_code == 200
    assert response.json["ok"] is False
    assert "Robot not connected" in response.json["error"]
    
    client.post("/robot/hardware_enable", json={"enable": False})

def test_robot_move_success(client, mock_ble):
    client.post("/robot/hardware_enable", json={"enable": True})
    MockClient, MockScanner = mock_ble
    
    # Force connected state
    client.post("/robot/connect", json={"address": "00:11:22:33:44:55"})
    
    response = client.post("/robot/move", json={"positions": {"1": 500}, "move_time": 1000})
    assert response.status_code == 200
    if not response.json["ok"]:
        print(response.json["error"])
    assert response.json["ok"] is True
    assert "completion unverified" in response.json["message"]
    
    # Verify write was called
    mock_instance = MockClient.return_value
    assert mock_instance.write_gatt_char.called
    
    # Cleanup
    client.get("/robot/disconnect")
    client.post("/robot/hardware_enable", json={"enable": False})

def test_robot_duplicate_command_prevention(client, mock_ble):
    client.post("/robot/hardware_enable", json={"enable": True})
    client.post("/robot/connect", json={"address": "00:11:22:33:44:55"})
    
    robot.busy = True # Simulate busy
    
    response = client.get("/robot/run/happy")
    # In app.py trigger_robot_for_emotion runs in a background thread and simply skips if busy
    # It returns ok=True but doesn't do anything. We check if busy prevented action by seeing if last_emotion_executed didn't update to new
    robot.last_emotion_executed = None
    import time
    time.sleep(0.1) # allow thread to run
    
    assert robot.last_emotion_executed is None
    
    robot.busy = False
    client.get("/robot/disconnect")
    client.post("/robot/hardware_enable", json={"enable": False})

def test_robot_stop_command(client, mock_ble):
    client.post("/robot/hardware_enable", json={"enable": True})
    client.post("/robot/connect", json={"address": "00:11:22:33:44:55"})
    MockClient, MockScanner = mock_ble
    
    response = client.get("/robot/stop")
    assert response.status_code == 200
    if not response.json["ok"]:
        print(response.json["error"])
    assert response.json["ok"] is True
    
    # Verify write was called for stop
    mock_instance = MockClient.return_value
    assert mock_instance.write_gatt_char.called
    
    client.get("/robot/disconnect")
    client.post("/robot/hardware_enable", json={"enable": False})

def test_emotion_to_pose_mapping(client):
    client.post("/robot/hardware_enable", json={"enable": False})
    for emotion in EMOTION_TO_POSE.keys():
        response = client.get(f"/robot/run/{emotion}")
        assert response.status_code == 200
        assert response.json["ok"] is True
        
    # Test invalid
    response = client.get("/robot/run/unknown_emotion_123")
    assert response.status_code == 200
    assert response.json["ok"] is False
