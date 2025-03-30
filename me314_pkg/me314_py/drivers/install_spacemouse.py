#!/usr/bin/env python3

import os
import subprocess

# Path to udev rules file
UDEV_RULES_FILENAME = "/etc/udev/rules.d/99-spacemouse.rules"

# Vendor and Product ID for your device (replace with your actual IDs if needed)
VENDOR_ID = "0x256f"
PRODUCT_ID = "0xc62e"

# The contents of the rule
rule = f'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{{idVendor}}=="{VENDOR_ID}", ATTRS{{idProduct}}=="{PRODUCT_ID}", MODE="0666"\n'

def main():
    # Check if script is run as root
    if os.geteuid() != 0:
        print("This script must be run as root (e.g. sudo) to create udev rules.")
        return

    # Write the rule to the file
    print(f"Writing the following rule to {UDEV_RULES_FILENAME}:")
    print(rule.strip())
    with open(UDEV_RULES_FILENAME, "w") as f:
        f.write(rule)

    # Reload udev rules
    print("Reloading udev rules...")
    subprocess.run(["udevadm", "control", "--reload-rules"], check=True)

    # Trigger so that existing devices are picked up
    print("Triggering new rules...")
    subprocess.run(["udevadm", "trigger"], check=True)

    print("Done. Please replug your SpaceMouse to apply the new permissions.")
    print("You can verify by running: ls -l /dev/hidraw*")
    print("The device for your SpaceMouse should have 'rw-rw-rw-' (0666) permissions now.")

if __name__ == "__main__":
    main()
