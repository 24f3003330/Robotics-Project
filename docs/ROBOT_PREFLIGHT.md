# Physical Robot Pre-Flight Verification Checklist

Before enabling `HARDWARE_ENABLED` or running physical commands (Stage C), the following must be verified against the physical robot. The application must remain in **Simulation only** mode until these items are confirmed.

## 1. Hardware Identification
- [ ] **Exact Model Name/Number:** Verify the robot is a LewanSoul/Hiwonder xArm (or explicitly document the correct model). The current implementation uses the `0x55 0x55` 6-DOF serial protocol. If it is a UFACTORY xArm, the integration will fail and may cause undefined behavior.
- [ ] **BLE Name / MAC Address:** Identify the expected BLE advertisement name and MAC address to avoid connecting to unrelated Bluetooth devices.

## 2. Capability Checks (Read Support)
- [ ] **Read Position Command:** Ascertain whether the robot's BLE controller supports reading servo positions back via BLE characteristic read/notify (e.g., command `0x15` in the LewanSoul protocol). 
- [ ] If read is not supported over BLE, be aware that command completion is **unverified** and relies strictly on static time estimates (`MOVE_TIME` + buffer).

## 3. Physical Limitations & Safety Bounds
- [ ] **Valid Servo IDs:** Confirm the robot uses exactly 6 servos mapped as IDs 1-6.
- [ ] **Servo Value Ranges:** The software currently validates bounds at `[0, 1000]`. Verify if your specific hardware limits are narrower (e.g., minimum 100, maximum 900).
- [ ] **Base and Extension Limits:** Identify any coordinates that cause the robot to collide with itself or the mounting surface. Update `move_servos` server-side validation to reject these specific regions if necessary.

## 4. Workcell Preparation
- [ ] **Safe Position (`front_safe`):** Physically test that the `front_safe` pose coordinates in `xarm_emotion_poses.json` represent a safe, neutral, non-colliding position.
- [ ] **Clear Workspace:** Ensure the sweep path of the arm is clear of people, cables, and fragile items.
- [ ] **Emergency Stop (E-Stop):** Ensure you have physical access to the power switch of the robot to kill power immediately if it behaves unexpectedly.

## 5. First Movement Sequence
1. Set the robot in an open area.
2. Open the application UI and verify it says **Simulation only**.
3. Use the **Robot BLE Control** section to click **Scan Robot**, select the verified MAC address, and click **Connect Robot**.
4. Enable hardware via API: `curl -X POST http://localhost:5000/robot/hardware_enable -H "Content-Type: application/json" -d '{"enable": true}'`
5. Click **Home**. Be prepared to hit the power switch if it moves erratically. 
6. Confirm the robot arrives at the `front_safe` position. Only after this succeeds should you test AI-triggered poses.
