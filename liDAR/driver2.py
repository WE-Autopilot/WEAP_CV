import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import struct
import math
import time
import json

# import the draw lidar bitmap functions
# from utils import lidar_to_bitmap

"""
    Provides functionality with CP210x LiDAR. Parses lidar packets and formats them.
    Courtesy of Hamza. Edits by Tygo.
"""

# Global buffer for incoming serial data
serial_buffer = b""

# Global accumulators for a block of 200 packets
block_packet_count = 0
block_angles_rad = []
block_distances = []

# Pre-calculate conversion factor (degrees to radians)
RAD_FACTOR = math.pi / 180.0

# Make a structure to hold run info, current format: {distance, angle}
# hold on, lidar_to_bitmap only accepts list[float]?
run_info = np.array([], dtype=[('distance', float), ('angle', float)])

def parse_packet(packet48: bytes):

    """
    Parse one 48-byte packet:
      1 byte  - header       (0x54)
      1 byte  - verLen       (0x2C)
      2 bytes - speed
      2 bytes - start_angle
      36 bytes - measurement data (12 x 3 bytes)
      2 bytes - end_angle
      2 bytes - timestamp
      2 bytes - crc
    """
    
    if len(packet48) != 48:
        raise ValueError("Packet must be exactly 48 bytes.")

    header, ver_len, speed, start_angle = struct.unpack("<BBHH", packet48[:6])
    data_bytes = packet48[6:42]
    end_angle, timestamp, crc = struct.unpack("<HHH", packet48[42:48])

    measurements = []
    for i in range(0, 36, 3):
        d0, d1, strength = data_bytes[i : i + 3]
        # Decode distance: combine low byte and lower 6 bits of high byte.
        distance = ((d1 & 0x3F) << 8) | d0
        invalid_flag = (d1 & 0x80) >> 7
        warning_flag = (d1 & 0x40) >> 6

        measurements.append({
            "distance": distance,
            "strength": strength,
            "invalid": invalid_flag,
            "warning": warning_flag
        })

    return {
        "header": header,
        "ver_len": ver_len,
        "speed": speed,
        "start_angle": start_angle,   # in hundredths of a degree
        "measurements": measurements,
        "end_angle": end_angle,       # in hundredths of a degree
        "timestamp": timestamp,
        "crc": crc
    }

def process_serial_data(ser):
    """
    Read all available data from the serial port and process complete packets.
    Returns the number of packets processed.
    """
    global serial_buffer, block_packet_count, block_angles_rad, block_distances

    packets_processed = 0
    # Read available data (up to 1024 bytes)
    data = ser.read(1024)
    if data:
        serial_buffer += data

    # Process all complete packets in the buffer.
    while True:
        # Look for the header bytes: 0x54, 0x2C.
        idx = serial_buffer.find(b"\x54\x2C")
        if idx < 0:
            break  # No header found.
        if idx + 48 > len(serial_buffer):
            break  # Not enough data for a complete packet.
        
        packet_bytes = serial_buffer[idx : idx + 48]
        serial_buffer = serial_buffer[idx + 48 :]
        try:
            packet = parse_packet(packet_bytes)
            #print(packet)
        except ValueError:
            continue

        # Convert start and end angles (in hundredths of a degree) to degrees.
        start_angle_deg = packet["start_angle"] / 100.0
        end_angle_deg = packet["end_angle"] / 100.0
        angle_diff = end_angle_deg - start_angle_deg
        if angle_diff < 0:
            angle_diff += 360.0

        num_meas = len(packet["measurements"])
        for i, meas in enumerate(packet["measurements"]):
            frac = i / (num_meas - 1) if num_meas > 1 else 0
            angle_deg = start_angle_deg + angle_diff * frac
            angle_rad = angle_deg * RAD_FACTOR  # convert to radians
            distance = meas["distance"] if not meas["invalid"] else 0

            block_angles_rad.append(angle_rad)
            block_distances.append(distance)

        block_packet_count += 1
        packets_processed += 1

        # run_info.append([distance, angle_deg])
        # print(f"Measurement {i}: Angle = {angle_deg:.2f}° ({angle_rad:.2f} rad) -> Distance = {distance}")

    return packets_processed

# Serial port settings, change depending on ur configuration (I think based on operating system)
PORT = "/dev/ttyUSB0"
BAUD = 230400
ser = serial.Serial(PORT, baudrate=BAUD, bytesize=8,
                    stopbits=serial.STOPBITS_ONE, parity=serial.PARITY_NONE,
                    timeout=0.01)

# Set up the polar plot (sonar-like view)
fig = plt.figure()
ax = plt.subplot(111, polar=True)
(line,) = ax.plot([], [], 'ro', markersize=1)
ax.set_rmax(4000)  # Adjust as needed for your sensor's range
ax.grid(True)
ax.set_title("Real-Time LIDAR Data (Block of 200 Packets)")

# Testing with draw bitmap functions
# bitmap = lidar_to_bitmap._lidar_to_bitmap()

def init():
    line.set_data([], [])
    return (line,)

def update(frame):
    global block_packet_count, block_angles_rad, block_distances
    # Process all available serial data.
    process_serial_data(ser)

    # Update the plot when 200 packets have been accumulated.
    if block_packet_count >= 20:
        line.set_data(block_angles_rad, block_distances)
        ax.set_title(f"Displayed Block of 200 Packets, Total Points: {len(block_angles_rad)}")
        # Reset accumulators for the next block.
        block_packet_count = 0
        block_angles_rad = []
        block_distances = []

    return (line,)

if __name__ == "__main__":
    # Create the animation: update every 50 ms.
    ani = animation.FuncAnimation(fig, update, init_func=init, interval=5, blit=True)

    plt.show()
    ser.close()
