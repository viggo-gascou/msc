#!/bin/bash
# Usage: ./submit.sh [sync|train|sync-train|run] [extra sbatch args]
# Example: ./submit.sh train --nodelist=cn13
#          ./submit.sh sync-train --nodelist=cn13 --time=4:00:00
#          ./submit.sh run <script> [extra sbatch args]

MODE="$1"
shift
# remaining args passed to sbatch call

mkdir -p logs/SLURM

# Get the repo root
# realpath resolves BASH_SOURCE[0] to the absolute path (the script), dirname strips the last element in the path
REPO_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

submit_job() {
    # if mode sync or sync-train git pull
    if [[ "$MODE" == "sync" || "$MODE" == "sync-train" ]]; then
        git pull
    fi
    LAST_JOB_ID=$(sbatch --parsable --chdir="$REPO_ROOT" "$@")
    echo "Submitted $MODE job: $LAST_JOB_ID" >&2
}

case "$MODE" in
    sync)
        submit_job jobs/sync.job
        ;;
    train)
        submit_job "$@" jobs/train.job
        ;;
    sync-train)
        submit_job jobs/sync.job
        SYNC_ID=$LAST_JOB_ID
        submit_job --dependency=afterok:$SYNC_ID "$@" jobs/train.job
        ;;
    run)
        SCRIPT=$1
        shift
        submit_job "$@" jobs/run.job "$SCRIPT"
        ;;
    sync-run)
        SCRIPT=$1
        shift
        submit_job jobs/sync.job
        SYNC_ID=$LAST_JOB_ID
        submit_job --dependency=afterok:$SYNC_ID "$@" jobs/run.job "$SCRIPT"
        ;;
    *)
        echo "Usage: $0 [sync|train|sync-train|run] [extra sbatch args]"
        exit 1
        ;;
esac
