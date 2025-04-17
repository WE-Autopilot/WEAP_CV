import driver   
import math
from threading import Lock

# one slot per integer degree
_NUM_DEGREES = 360
scan_buffer = [0] * _NUM_DEGREES
_scan_lock   = Lock()

def update_scan():
    """
    Pull every parsed packet waiting in driver.packet_queue,
    map its 12 measurements into scan_buffer[0..359], rounding
    to the nearest degree. Overwrites with newest distance.
    """

    while not driver.packet_queue.empty():
        pkt = driver.packet_queue.get()

        # Convert the hundredths-of-a-degree fields into float degrees
        start_deg = pkt["start_angle"] / 100.0
        end_deg   = pkt["end_angle"]   / 100.0

        # List of measurements and how wide this sweep is
        meas_list = pkt["measurements"]
        span      = (end_deg - start_deg) % 360.0
        count     = len(meas_list)

        # Lock the shared scan_buffer so no other thread writes at the same time
        with _scan_lock:
            for idx, measurement in enumerate(meas_list):
                # Figure out where along the sweep this point lies (0.0→1.0)
                fraction = idx / (count - 1) if count > 1 else 0

                # Compute the actual absolute angle (and wrap if >360)
                angle = (start_deg + span * fraction) % 360.0

                # Round to the nearest integer degree slot
                degree_index = int(round(angle)) % 360

                # Store the distance (0 if flagged invalid)
                if measurement["invalid"]:
                    scan_buffer[degree_index] = 0
                else:
                    scan_buffer[degree_index] = measurement["distance"]

def get_latest_scan():
    """
    Atomically grab your current 0–359° distance array.
    """
    with _scan_lock:
        return scan_buffer.copy()

if __name__ == "__main__":
    # start the driver background reader
    driver.start_reader()

    # example loop: once per second, fold in new packets, then print the full 360° list
    import time
    while True:
        update_scan()
        scan = get_latest_scan()
        # this will be a list of 360 distances, index 0→0°, …, index 359→359°
        print(scan)
        time.sleep(1.0)

