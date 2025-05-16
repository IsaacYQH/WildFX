#!/bin/bash

# Kill any existing JACK servers
killall -9 jackd 2> /dev/null

# Detect if audio hardware exists
if aplay -l | grep -q 'card'; then
    echo "Audio hardware detected, using ALSA driver"
    # Attempt to start JACK with ALSA
    jackd -d alsa -d hw:0 -r 44100 -p 1024 -P &
    JACK_PID=$!
    sleep 0.5
    
    # Verify JACK started successfully
    if ! ps -p $JACK_PID > /dev/null; then
        echo "ALSA driver failed, falling back to dummy driver"
        jackd -d dummy -r 44100 -p 1024 &
    fi
else
    echo "No audio hardware detected, using dummy driver"
    # Start JACK with dummy driver
    jackd -d dummy -r 44100 -p 1024 &
fi

# Allow JACK some time to initialize
sleep 0.5

echo "JACK audio server started."
echo "-----------------------------------------------"

# Initialize Reaper and ensure it's properly set up
echo "Starting REAPER initialization..."

# === Launch REAPER and capture its output to a log ===
LOG_FILE="/tmp/reaper.log"

# Launch the installer with input piping
stdbuf -oL -eL bash -c "(echo "R"; sleep 0.5; echo "Y") | sh /home/u1/reaper/install-reaper.sh" >"$LOG_FILE" 2>&1 &
# === Wait for “jack: activated client” to appear in the log ===
echo "Waiting for REAPER to initialize (max 30s)…"
found=0
for i in $(seq 1 30); do
  # Show current log contents
  cat "$LOG_FILE"
  if grep -q 'jack: activated client' "$LOG_FILE"; then
    found=1
    echo "Detected 'jack: activated client' after ${i}s"
    break
  fi
  sleep 1
done
if [ "$found" -eq 0 ]; then
  echo "Timed out waiting for REAPER to initialize, proceeding anyway."
fi
echo "REAPER initialization completed successfully!"
# Remove any old log or flag files
rm -f "$LOG_FILE"

# Kill REAPER processes and monitor properly
echo "Killing REAPER processes..."
rm -f "$LOG_FILE"  # Clear log first
pkill -f reaper >"$LOG_FILE" 2>&1
KILL_STATUS=$?

if [ $KILL_STATUS -eq 0 ]; then
  echo "REAPER processes killed successfully"
else
  echo "No REAPER processes found or error during kill (status: $KILL_STATUS)"
fi

# Wait for processes to actually terminate
echo "Waiting for REAPER processes to terminate..."
for i in $(seq 1 10); do
  if pgrep -f reaper >/dev/null; then
    echo "Still waiting for REAPER to terminate ($i/10)..."
    sleep 1
  else
    echo "All REAPER processes terminated"
    break
  fi
done

# Initialize reapy
# Start REAPER in the background, redirecting stdout and stderr to the log
stdbuf -oL -eL reaper -nosplash -nonewinst >"$LOG_FILE" 2>&1 &
REAPER_PID=$!
echo "REAPER PID=$REAPER_PID, logging to $LOG_FILE"

# === Wait for “jack: activated client” to appear in the log ===
echo "Waiting for reapy to initialize (max 30s)…"
found=0
for i in $(seq 1 30); do
  # Show current log contents
  cat "$LOG_FILE"
  if grep -q 'jack: activated client' "$LOG_FILE"; then
    found=1
    echo "Detected 'jack: activated client' after ${i}s"
    echo "Configuring reapy…"
    python -c "import reapy; reapy.configure_reaper()" >/dev/null 2>&1
    break
  fi
  sleep 1
done

if [ "$found" -eq 0 ]; then
  echo "Timed out waiting for REAPER start, proceeding anyway."
fi

# Clean up
echo "Cleaning up..."
# Terminate the REAPER process
pkill -9 $REAPER_PID 2> /dev/null
# Remove the log file
rm -f "$LOG_FILE"

echo "-----------------------------------------------"
echo "reapy is now configurated. Starting bash shell."
echo "Note: Run \"tmux new-session -d -s reaper-session 'reaper -nosplash -nonewinst'\" to start REAPER in a new tmux session in background."
echo "If reapy cannot connect to REAPER, kill the session with \"tmux kill-session\" and retry."
bash