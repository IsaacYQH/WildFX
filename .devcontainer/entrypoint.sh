#!/usr/bin/env bash
set -Eeuo pipefail

export DISPLAY="${DISPLAY:-:99}"
export JACK_NO_AUDIO_RESERVATION=1
REAPER_LOG=/tmp/wildfx-reaper.log
REAPY_MARKER=/home/u1/.wildfx-reapy-configured

copy_plugins() {
    local plugin_format
    for plugin_format in .vst3 .vst .clap .lv2; do
        mkdir -p "/home/u1/${plugin_format}"
        if [[ -d "/plugins-host/${plugin_format}" ]]; then
            rsync -a --no-owner --no-group --exclude='__MACOSX/' \
                --exclude='._*' \
                "/plugins-host/${plugin_format}/" "/home/u1/${plugin_format}/"
        fi
    done
}

start_reaper() {
    : >"${REAPER_LOG}"
    stdbuf -oL -eL reaper -nosplash -nonewinst -noactivate \
        >"${REAPER_LOG}" 2>&1 &
    REAPER_PID=$!
}

wait_for_reaper_process() {
    local attempt
    for attempt in $(seq 1 3); do
        sleep 1
        if ! kill -0 "${REAPER_PID}" 2>/dev/null; then
            echo "REAPER exited during startup. Log follows:" >&2
            sed -n '1,200p' "${REAPER_LOG}" >&2
            return 1
        fi
    done
    return 0
}

wait_for_reaper_scan() {
    local attempt
    local stable_seconds=0
    for attempt in $(seq 1 180); do
        if ! kill -0 "${REAPER_PID}" 2>/dev/null; then
            echo "REAPER exited while scanning plugins. Log follows:" >&2
            sed -n '1,200p' "${REAPER_LOG}" >&2
            return 1
        fi

        # REAPER launches child processes with -__vst_scan__ while it builds
        # its plugin cache. Short gaps can occur between plugins, so require a
        # stable quiet period before editing or using the control interface.
        if pgrep -f -- '-__vst_scan__' >/dev/null 2>&1; then
            stable_seconds=0
        else
            stable_seconds=$((stable_seconds + 1))
            if [[ "${stable_seconds}" -ge 5 ]]; then
                return 0
            fi
        fi
        sleep 1
    done

    echo "REAPER plugin scanning did not settle within 180 seconds." >&2
    sed -n '1,200p' "${REAPER_LOG}" >&2
    return 1
}

copy_plugins

if ! pgrep -f 'Xvfb :99' >/dev/null 2>&1; then
    # Docker restart preserves /tmp in the container, including stale X locks.
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
    Xvfb :99 -screen 0 1280x720x24 >/tmp/wildfx-xvfb.log 2>&1 &
fi
xvfb_ready=0
for attempt in $(seq 1 20); do
    if [[ -S /tmp/.X11-unix/X99 ]]; then
        xvfb_ready=1
        break
    fi
    sleep 0.25
done
if [[ "${xvfb_ready}" -ne 1 ]]; then
    echo "Xvfb did not become ready. Log follows:" >&2
    sed -n '1,200p' /tmp/wildfx-xvfb.log >&2
    exit 1
fi
if ! pgrep -x jackd >/dev/null 2>&1; then
    # Offline rendering does not need realtime scheduling privileges.
    jackd --no-realtime -d dummy -r 44100 -p 1024 \
        >/tmp/wildfx-jack.log 2>&1 &
fi

jack_ready=0
for attempt in $(seq 1 30); do
    if jack_lsp >/dev/null 2>&1; then
        jack_ready=1
        break
    fi
    sleep 1
done
if [[ "${jack_ready}" -ne 1 ]]; then
    echo "JACK did not become ready. Log follows:" >&2
    sed -n '1,200p' /tmp/wildfx-jack.log >&2
    exit 1
fi

start_reaper
wait_for_reaper_process
wait_for_reaper_scan

if [[ ! -f "${REAPY_MARKER}" ]]; then
    configured=0
    for attempt in $(seq 1 6); do
        if timeout 10s python -c 'import reapy; reapy.configure_reaper()' \
            >/tmp/wildfx-reapy-configure.log 2>&1; then
            configured=1
            touch "${REAPY_MARKER}"
            break
        fi
        sleep 1
    done
    if [[ "${configured}" -ne 1 ]]; then
        echo "Could not configure reapy. Log follows:" >&2
        sed -n '1,200p' /tmp/wildfx-reapy-configure.log >&2
        exit 1
    fi

    # The first process loaded the pre-configuration state. Restart exactly
    # that PID so REAPER loads the reapy startup hook.
    kill "${REAPER_PID}" 2>/dev/null || true
    wait "${REAPER_PID}" 2>/dev/null || true
    start_reaper
    wait_for_reaper_process
    wait_for_reaper_scan
fi

connected=0
for attempt in $(seq 1 30); do
    if timeout 3s python -c 'import reapy; reapy.Project().id' \
        >/dev/null 2>&1; then
        connected=1
        break
    fi
    sleep 1
done
if [[ "${connected}" -ne 1 ]]; then
    echo "REAPER is running, but reapy did not accept a connection." >&2
    sed -n '1,200p' "${REAPER_LOG}" >&2
    exit 1
fi

echo "WildFX ready: JACK, REAPER, and reapy are running."
exec "$@"
