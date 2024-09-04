from resensepy import sensor
from time import sleep
import numpy as np

def calibrate(values) -> float:
    return np.mean(values)

def main():
    # Init sensor
    sensor_instance = sensor.HEXSensor('/dev/ttyACM0')
    sample_rate = 100
    print_frequency = 100

    if not sensor_instance.connect():
        print("Failed to connect to sensor.")
        raise Exception("Could not connect to force sensor")

    # Continuously print out force sensor data to terminal
    calibration_samples = []
    for _ in range(1000):
        s = sensor_instance.record_sample()
        calibration_samples.append(s.force.z)
        sleep(1/sample_rate)

    z = calibrate(calibration_samples)
    print(f"Calibrated z force: {z:.2f}N")
    sleep(2)

    while True:
        sample = sensor_instance.record_sample()
        print(f"Z force: {sample.force.z - z:.2f}")
        sleep(1 / print_frequency)

if __name__ == "__main__":
    main()