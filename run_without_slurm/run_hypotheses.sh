#!/usr/bin/env bash

set -uo pipefail

CONFIG_ARG="${1:-parameterization_configs/h1_configs}"
SITE="${2-}"
MIN_CPUS_PER_JOB="${TUSSOCK_MIN_CPUS_PER_JOB:-10}"
NOTIFY_EMAIL="${TUSSOCK_NOTIFY_EMAIL:-}"
SECONDS=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$PWD/scripts/parameterization.py" ]]; then
    ROOT_DIR="$PWD"
elif [[ -f "$SCRIPT_DIR/../scripts/parameterization.py" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    echo "ERROR: Could not find scripts/parameterization.py."
    exit 1
fi

PARAM_SCRIPT="$ROOT_DIR/scripts/parameterization.py"

if [[ "$CONFIG_ARG" = /* ]]; then
    CONFIG_DIR="$CONFIG_ARG"
elif [[ -d "$PWD/$CONFIG_ARG" ]]; then
    CONFIG_DIR="$(cd "$PWD/$CONFIG_ARG" && pwd)"
elif [[ -d "$ROOT_DIR/$CONFIG_ARG" ]]; then
    CONFIG_DIR="$(cd "$ROOT_DIR/$CONFIG_ARG" && pwd)"
else
    echo "ERROR: Configuration directory not found: $CONFIG_ARG"
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: Could not find python or python3."
    exit 1
fi

if ! [[ "$MIN_CPUS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TUSSOCK_MIN_CPUS_PER_JOB must be a positive integer."
    exit 1
fi

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "ERROR: Bash 5.1 or newer is required."
    echo "Current version: $BASH_VERSION"
    exit 1
fi

shopt -s nullglob
CONFIG_FILES=("$CONFIG_DIR"/*.ini)
shopt -u nullglob

NUM_CONDITIONS="${#CONFIG_FILES[@]}"

if (( NUM_CONDITIONS == 0 )); then
    echo "ERROR: No .ini files found in: $CONFIG_DIR"
    exit 1
fi

CPU_IDS=()

expand_cpu_list() {
    local spec="$1"
    local chunk
    local start
    local end
    local cpu

    spec="${spec//[[:space:]]/}"
    IFS=',' read -r -a chunks <<< "$spec"

    for chunk in "${chunks[@]}"; do
        if [[ "$chunk" == *-* ]]; then
            start="${chunk%-*}"
            end="${chunk#*-}"

            for ((cpu = start; cpu <= end; cpu++)); do
                CPU_IDS+=("$cpu")
            done
        elif [[ -n "$chunk" ]]; then
            CPU_IDS+=("$chunk")
        fi
    done
}

if command -v taskset >/dev/null 2>&1; then
    CPU_SPEC="$(
        taskset -pc "$$" 2>/dev/null |
        sed -E 's/^.*:[[:space:]]*//'
    )"

    if [[ -n "$CPU_SPEC" ]]; then
        expand_cpu_list "$CPU_SPEC"
    fi
fi

if (( ${#CPU_IDS[@]} == 0 )); then
    if command -v nproc >/dev/null 2>&1; then
        NUM_DETECTED_CPUS="$(nproc)"
    else
        NUM_DETECTED_CPUS="$(
            "$PYTHON_BIN" -c 'import os; print(os.cpu_count() or 1)'
        )"
    fi

    for ((cpu = 0; cpu < NUM_DETECTED_CPUS; cpu++)); do
        CPU_IDS+=("$cpu")
    done
fi

NUM_CPUS="${#CPU_IDS[@]}"

if [[ -n "${TUSSOCK_TOTAL_CPUS:-}" ]]; then
    if ! [[ "$TUSSOCK_TOTAL_CPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TUSSOCK_TOTAL_CPUS must be a positive integer."
        exit 1
    fi

    if (( TUSSOCK_TOTAL_CPUS < NUM_CPUS )); then
        CPU_IDS=("${CPU_IDS[@]:0:TUSSOCK_TOTAL_CPUS}")
        NUM_CPUS="${#CPU_IDS[@]}"
    fi
fi

if (( NUM_CPUS < MIN_CPUS_PER_JOB )); then
    echo "ERROR: $NUM_CPUS CPUs are available, but each job requires at least $MIN_CPUS_PER_JOB."
    exit 1
fi

MAX_SLOTS=$((NUM_CPUS / MIN_CPUS_PER_JOB))
NUM_SLOTS="$NUM_CONDITIONS"

if (( MAX_SLOTS < NUM_SLOTS )); then
    NUM_SLOTS="$MAX_SLOTS"
fi

if [[ -n "${TUSSOCK_MAX_WORKERS:-}" ]]; then
    if ! [[ "$TUSSOCK_MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TUSSOCK_MAX_WORKERS must be a positive integer."
        exit 1
    fi

    if (( TUSSOCK_MAX_WORKERS < NUM_SLOTS )); then
        NUM_SLOTS="$TUSSOCK_MAX_WORKERS"
    fi
fi

BASE_CPUS=$((NUM_CPUS / NUM_SLOTS))
EXTRA_CPUS=$((NUM_CPUS % NUM_SLOTS))

join_by_comma() {
    local IFS=','
    echo "$*"
}

SLOT_CPU_LISTS=()
SLOT_CPU_COUNTS=()

cpu_offset=0

for ((slot = 0; slot < NUM_SLOTS; slot++)); do
    count="$BASE_CPUS"

    if (( slot < EXTRA_CPUS )); then
        count=$((count + 1))
    fi

    ids=()

    for ((j = 0; j < count; j++)); do
        ids+=("${CPU_IDS[$cpu_offset]}")
        cpu_offset=$((cpu_offset + 1))
    done

    SLOT_CPU_LISTS[$slot]="$(join_by_comma "${ids[@]}")"
    SLOT_CPU_COUNTS[$slot]="$count"
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT_DIR/local_hypothesis_logs/$TIMESTAMP"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1

send_notification() {
    local result="$1"
    local exit_code="$2"
    local host
    local duration
    local subject
    local body

    host="$(hostname -f 2>/dev/null || hostname)"
    duration="$((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s"
    subject="[Tussock model] $result: $(basename "$CONFIG_DIR")"

    body="$(
        cat <<EOF
Tussock-model hypothesis run finished.

Result: $result
Exit code: $exit_code
Machine: $host
Configuration directory: $CONFIG_DIR
Hypotheses: $NUM_CONDITIONS
CPUs used: $NUM_CPUS
Concurrent jobs: $NUM_SLOTS
Minimum CPUs per job: $MIN_CPUS_PER_JOB
Duration: $duration
Logs: $LOG_DIR
Finished: $(date '+%Y-%m-%d %H:%M:%S')
EOF
    )"

    if command -v mail >/dev/null 2>&1; then
        printf '%s\n' "$body" |
            mail -s "$subject" "$NOTIFY_EMAIL"
    elif command -v mailx >/dev/null 2>&1; then
        printf '%s\n' "$body" |
            mailx -s "$subject" "$NOTIFY_EMAIL"
    elif command -v msmtp >/dev/null 2>&1; then
        {
            printf 'To: %s\n' "$NOTIFY_EMAIL"
            printf 'Subject: %s\n' "$subject"
            printf '\n%s\n' "$body"
        } | msmtp "$NOTIFY_EMAIL"
    elif command -v sendmail >/dev/null 2>&1; then
        {
            printf 'To: %s\n' "$NOTIFY_EMAIL"
            printf 'Subject: %s\n' "$subject"
            printf '\n%s\n' "$body"
        } | sendmail -t
    else
        echo "WARNING: Email notification could not be sent."
        echo "Install and configure mail, mailx, msmtp, or sendmail."
        return 1
    fi

    echo "Notification sent to $NOTIFY_EMAIL"
}

if [[ -n "$SITE" ]]; then
    SITE_DISPLAY="$SITE"
else
    SITE_DISPLAY="[all sites]"
fi

echo "============================================================"
echo "Config directory:   $CONFIG_DIR"
echo "Parameter script:   $PARAM_SCRIPT"
echo "Python:              $PYTHON_BIN"
echo "Site:                $SITE_DISPLAY"
echo "Hypotheses:          $NUM_CONDITIONS"
echo "Available CPUs:      $NUM_CPUS"
echo "Minimum CPUs/job:    $MIN_CPUS_PER_JOB"
echo "Concurrent jobs:     $NUM_SLOTS"
echo "Queued jobs:         $((NUM_CONDITIONS - NUM_SLOTS))"
echo "Notification email:  $NOTIFY_EMAIL"
echo "Logs:                $LOG_DIR"
echo "============================================================"

for ((slot = 0; slot < NUM_SLOTS; slot++)); do
    echo "Slot $((slot + 1)): ${SLOT_CPU_COUNTS[$slot]} CPUs (${SLOT_CPU_LISTS[$slot]})"
done

echo "============================================================"

run_one() {
    local slot="$1"
    local config_index="$2"
    local cpu_list="${SLOT_CPU_LISTS[$slot]}"
    local cpu_count="${SLOT_CPU_COUNTS[$slot]}"
    local cfg="${CONFIG_FILES[$config_index]}"
    local name
    local run_name
    local run_log
    local status_file
    local rc

    name="$(basename "$cfg" .ini)"
    run_name="$(printf '%03d_%s' "$((config_index + 1))" "$name")"
    run_log="$LOG_DIR/${run_name}.log"
    status_file="$LOG_DIR/${run_name}.status"

    CMD=(
        "$PYTHON_BIN"
        "$PARAM_SCRIPT"
        --config "$cfg"
    )

    if [[ -n "$SITE" ]]; then
        CMD+=(--site "$SITE")
    fi

    {
        echo "status=RUNNING"
        echo "config=$cfg"
        echo "site=${SITE:-ALL}"
        echo "slot=$((slot + 1))"
        echo "cpu_list=$cpu_list"
        echo "cpu_count=$cpu_count"
    } > "$status_file"

    echo "Starting $run_name on slot $((slot + 1)) with $cpu_count CPUs"

    if command -v taskset >/dev/null 2>&1; then
        taskset -c "$cpu_list" \
            env \
            PYTHONUNBUFFERED=1 \
            TUSSOCK_NUM_THREADS="$cpu_count" \
            SLURM_CPUS_PER_TASK="$cpu_count" \
            OMP_NUM_THREADS="$cpu_count" \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 \
            "${CMD[@]}" \
            2>&1 |
            sed -u "s/^/[$run_name] /" |
            tee "$run_log"

        rc=${PIPESTATUS[0]}
    else
        env \
            PYTHONUNBUFFERED=1 \
            TUSSOCK_NUM_THREADS="$cpu_count" \
            SLURM_CPUS_PER_TASK="$cpu_count" \
            OMP_NUM_THREADS="$cpu_count" \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 \
            "${CMD[@]}" \
            2>&1 |
            sed -u "s/^/[$run_name] /" |
            tee "$run_log"

        rc=${PIPESTATUS[0]}
    fi

    if (( rc == 0 )); then
        {
            echo "status=COMPLETE"
            echo "exit_code=0"
            echo "config=$cfg"
            echo "slot=$((slot + 1))"
            echo "cpu_list=$cpu_list"
            echo "cpu_count=$cpu_count"
        } > "$status_file"

        echo "Completed $run_name"
    else
        {
            echo "status=FAILED"
            echo "exit_code=$rc"
            echo "config=$cfg"
            echo "slot=$((slot + 1))"
            echo "cpu_list=$cpu_list"
            echo "cpu_count=$cpu_count"
            echo "log=$run_log"
        } > "$status_file"

        echo "FAILED $run_name with exit code $rc"
    fi

    return "$rc"
}

SLOT_PIDS=()
declare -A PID_TO_SLOT

next_config=0
running=0
overall_status=0

cleanup() {
    echo "Stopping jobs..."

    for pid in "${SLOT_PIDS[@]:-}"; do
        if [[ -n "$pid" ]]; then
            kill "$pid" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true
    send_notification "INTERRUPTED" 130 || true
    exit 130
}

trap cleanup INT TERM

launch_in_slot() {
    local slot="$1"
    local config_index="$2"
    local pid

    run_one "$slot" "$config_index" &

    pid=$!
    SLOT_PIDS[$slot]="$pid"
    PID_TO_SLOT[$pid]="$slot"
    running=$((running + 1))
}

for ((slot = 0; slot < NUM_SLOTS && next_config < NUM_CONDITIONS; slot++)); do
    launch_in_slot "$slot" "$next_config"
    next_config=$((next_config + 1))
done

while (( running > 0 )); do
    finished_pid=""

    if wait -n -p finished_pid; then
        rc=0
    else
        rc=$?
        overall_status=1
    fi

    slot="${PID_TO_SLOT[$finished_pid]}"

    unset 'PID_TO_SLOT[$finished_pid]'
    SLOT_PIDS[$slot]=""
    running=$((running - 1))

    if (( next_config < NUM_CONDITIONS )); then
        launch_in_slot "$slot" "$next_config"
        next_config=$((next_config + 1))
    fi
done

trap - INT TERM

echo "============================================================"

if (( overall_status == 0 )); then
    RESULT="SUCCESS"
    echo "All hypotheses completed successfully."
else
    RESULT="FAILED"
    echo "One or more hypotheses failed."
fi

echo "Logs: $LOG_DIR"
echo "============================================================"

send_notification "$RESULT" "$overall_status" || true

exit "$overall_status"