#!/usr/bin/env python3
"""
Test script to verify REAPER and reapy are properly configured and working.
"""

import sys
import time
import subprocess
import os

def test_reaper_connection():
    print("Testing REAPER and reapy setup...")
    
    # Step 1: Try to import reapy
    try:
        import reapy
        print("✓ Successfully imported reapy")
    except ImportError as e:
        print(f"✗ Failed to import reapy: {str(e)}")
        print("  Make sure reapy is installed: pip install python-reapy")
        return False
    
    # Step 2: Try to configure REAPER
    try:
        reapy.configure_reaper()
        print("✓ Successfully configured REAPER")
    except Exception as e:
        print(f"✗ Failed to configure REAPER: {str(e)}")
        return False
    
    # Step 3: Try to connect to REAPER
    try:
        reapy.connect()
        print("✓ Successfully connected to REAPER")
    except Exception as e:
        print(f"✗ Failed to connect to REAPER: {str(e)}")
        print("  Attempting to start REAPER...")
        
        # Try to start REAPER
        try:
            # Look for REAPER in common locations or from environment variable
            reaper_path = os.environ.get("REAPER_PATH", "reaper")
            subprocess.Popen([reaper_path, "-nosplash"])
            print("  Started REAPER, waiting for it to initialize...")
            time.sleep(5)  # Wait for REAPER to start
            
            # Try connecting again
            try:
                reapy.connect()
                print("✓ Successfully connected to REAPER after starting it")
            except Exception as e2:
                print(f"✗ Still failed to connect to REAPER: {str(e2)}")
                return False
        except Exception as e3:
            print(f"✗ Failed to start REAPER: {str(e3)}")
            return False
    
    # Step 4: Try a basic REAPER operation
    try:
        project = reapy.Project()
        print(f"✓ Successfully accessed project: {project.name or 'Unnamed'}")
        return True
    except Exception as e:
        print(f"✗ Failed to access REAPER project: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_reaper_connection()
    if success:
        print("\n✅ REAPER and reapy are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ There were issues with REAPER and/or reapy setup.")
        sys.exit(1)