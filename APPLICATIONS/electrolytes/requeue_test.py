#!/usr/bin/env python3
import time
import signal
import sys

def sigterm_handler(signum, frame):
    """Handle SIGTERM gracefully"""
    print(f"Received SIGTERM (signal {signum})")
    print("Gracefully shutting down...")
    print("Exiting cleanly for requeue...")
    sys.exit(0)

def sigusr1_handler(signum, frame):
    """Handle USR1 signal (timeout warning)"""
    print(f"Received USR1 (signal {signum}) - timeout warning!")
    print("Will continue running until SIGTERM...")
    # Don't exit here - just log the warning

def main():
    # Set up signal handlers
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)
    
    print("Starting requeue test script...")
    print("This script will run and handle SIGTERM for testing requeue...")
    
    i = 0
    start_time = time.time()
    
    try:
        while True:
            i += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            if i % 50000 == 0:  # Print every ~5 seconds
                print(f"Running... iteration {i}, elapsed: {elapsed:.1f}s")
            
            # Small sleep to prevent 100% CPU usage
            time.sleep(0.0001)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()