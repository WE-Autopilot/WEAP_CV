# driver.py
import serial
import struct
from queue import Queue
from threading import Thread

# change based off how it's done in linux 
PORT      = "COM6"
BAUD      = 230400
PACKET_SZ = 48

# this is where parsed packets go after it's coming in from the port 
packet_queue = Queue()

def parse_packet(packet48: bytes) -> dict:
    """
    Parsing the packet into a dict, basically copied how the response looks based off the print statement.
    """
    if len(packet48) != PACKET_SZ:
        raise ValueError("Packet must be exactly 48 bytes.")
    header, ver_len, speed, start_angle = struct.unpack("<BBHH", packet48[:6])
    data_bytes = packet48[6:42]
    end_angle, timestamp, crc = struct.unpack("<HHH", packet48[42:48])

    measurements = []
    for i in range(0, 36, 3):
        d0, d1, strength = data_bytes[i : i + 3]
        distance = ((d1 & 0x3F) << 8) | d0
        invalid = (d1 & 0x80) >> 7
        warning = (d1 & 0x40) >> 6
        measurements.append({
            "distance": distance,
            "strength": strength,
            "invalid": invalid,
            "warning": warning
        })

    return {
        "header":     header,
        "ver_len":    ver_len,
        "speed":      speed,
        "start_angle": start_angle,     # hundredths of a degree
        "measurements": measurements,
        "end_angle":  end_angle,        # hundredths of a degree
        "timestamp":  timestamp,
        "crc":        crc
    }

def _reader_loop():
    # We use a small timeout so read() returns quickly even if there’s no data.
    ser = serial.Serial(
        PORT,
        baudrate=BAUD,
        bytesize=8,
        stopbits=serial.STOPBITS_ONE,
        parity=serial.PARITY_NONE,
        timeout=0.01
    )

    # This buffer will hold raw bytes until we can form complete 48‑byte packets.
    buf = bytearray()

    # Keep running forever in the background thread
    while True:
        # Read up to 1024 bytes from the serial port.
        chunk = ser.read(1024)
        if chunk:
            buf.extend(chunk)

        while True:
            idx = buf.find(b"\x54\x2C")

            # If we didn’t find the header or don’t have enough bytes
            # remaining for a full packet, stop parsing for now.
            if idx < 0 or len(buf) < idx + PACKET_SZ:
                break

            # Slice out exactly one packet’s worth of bytes.
            pkt = bytes(buf[idx : idx + PACKET_SZ])

            # Remove everything up through that packet from the buffer, so we don’t re‑parse it on the next loop.
            del buf[: idx + PACKET_SZ]

            # Try to parse the packet into a structured dict.
            try:
                parsed = parse_packet(pkt)
            except ValueError:
                # Bad packet—ignore and keep going
                continue

            # if this works, it will be sent to the util file
            packet_queue.put(parsed)


def start_reader():
    """
    Spin up the background thread that fills `packet_queue`.
    Safe to call multiple times (only one thread will run).
    """
    if not hasattr(start_reader, "_started"):
        start_reader._started = True
        t = Thread(target=_reader_loop, daemon=True)
        t.start()
