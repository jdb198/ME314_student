#!/usr/bin/env python3
# Used to check the HID devices connected to the computer.

import hid

def list_hid_devices():
    devices = hid.enumerate()
    
    if not devices:
        print("No HID devices found.")
        return

    for i, device in enumerate(devices, start=1):
        print(f"\nDevice {i}:")
        print(f"  Vendor ID      : {hex(device['vendor_id'])}")
        print(f"  Product ID     : {hex(device['product_id'])}")
        print(f"  Manufacturer   : {device.get('manufacturer_string', 'Unknown')}")
        print(f"  Product        : {device.get('product_string', 'Unknown')}")
        print(f"  Serial Number  : {device.get('serial_number', 'Unknown')}")
        print(f"  Usage Page     : {hex(device['usage_page'])}")
        print(f"  Usage          : {hex(device['usage'])}")
        print(f"  Path           : {device['path'].decode() if isinstance(device['path'], bytes) else device['path']}")
        print(f"  Interface      : {device.get('interface_number', 'Unknown')}")

if __name__ == "__main__":
    list_hid_devices()


# sudo chmod 666 /dev/hidraw10
