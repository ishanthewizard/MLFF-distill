import time
import signal
import sys
import os

def handle_sigterm(signum, frame):
    """Graceful shutdown handler for SIGTERM."""
    print(f"FAKE SIMULATION (PID: {os.getpid()}): Caught SIGTERM! Shutting down gracefully.", flush=True)
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGTERM, handle_sigterm)

print(f"FAKE SIMULATION (PID: {os.getpid()}): Starting up and running in a loop.", flush=True)
print("Send SIGTERM to my parent srun process to test preemption forwarding.", flush=True)
print("Send USR1 to the main batch script to test timeout handling.", flush=True)

# Run an infinite loop to keep the process alive
try:
    while True:
        print("sleeping for 10 seconds")
        time.sleep(10)
except KeyboardInterrupt:
    print(f"FAKE SIMULATION (PID: {os.getpid()}): Keyboard interrupt received. Exiting.", flush=True)
    sys.exit(0)
