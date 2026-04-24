#!/bin/bash
# Usage: ./submit.sh [sync|train|train-multi|sync-train|run] [extra sbatch args]
# Example: ./submit.sh train --nodelist=cn13
#          ./submit.sh train-multi
#          ./submit.sh sync-train --nodelist=cn13 --time=4:00:00
#          ./submit.sh run <script> [extra sbatch args]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [sync|train|train-multi|sync-train|sync-train-multi|run|sync-run] [extra sbatch args]" >&2
    exit 1
fi

MODE="$1"
shift
# remaining args passed to sbatch call

mkdir -p logs/SLURM

# Get the repo root
# realpath resolves BASH_SOURCE[0] to the absolute path (the script), dirname strips the last element in the path
REPO_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Run git pull ONCE for sync* modes
if [[ "$MODE" == sync* ]]; then
    git -C "$REPO_ROOT" pull
fi

submit_job() {
    local JOB_NAME="$1"
    shift

    LAST_JOB_ID=$(sbatch --parsable --chdir="$REPO_ROOT" "$@")
    echo "Submitted $JOB_NAME job: $LAST_JOB_ID" >&2
}

case "$MODE" in
    sync)
        submit_job sync jobs/sync.job
        ;;
    train)
        submit_job train "$@" jobs/train.job
        ;;
    train-multi)
        submit_job train-multi "$@" jobs/train-multi.job
        ;;
    sync-train-multi)
        submit_job sync jobs/sync.job
        SYNC_ID=$LAST_JOB_ID
        submit_job train-multi --dependency=afterok:$SYNC_ID "$@" jobs/train-multi.job
        ;;
    sync-train)
        submit_job sync jobs/sync.job
        SYNC_ID=$LAST_JOB_ID
        submit_job train --dependency=afterok:$SYNC_ID "$@" jobs/train.job
        ;;
    run)
        if [[ $# -lt 1 ]]; then
            echo "Error: run mode requires a script argument" >&2
            exit 1
        fi
        SCRIPT="$1"
        shift
        submit_job run "$@" jobs/run.job "$SCRIPT"
        ;;
    sync-run)
        if [[ $# -lt 1 ]]; then
            echo "Error: sync-run mode requires a script argument" >&2
            exit 1
        fi
        SCRIPT="$1"
        shift
        submit_job sync jobs/sync.job
        SYNC_ID=$LAST_JOB_ID
        submit_job run --dependency=afterok:$SYNC_ID "$@" jobs/run.job "$SCRIPT"
        ;;
    *)
        echo "Usage: $0 [sync|train|train-multi|sync-train|sync-train-multi|run|sync-run] [extra sbatch args]" >&2
        exit 1
        ;;
esac
