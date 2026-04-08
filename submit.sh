#!/bin/bash
# Usage: ./submit.sh [sync|train|sync-train] [extra sbatch args for train]
# Example: ./submit.sh train --nodelist=cn13
#          ./submit.sh sync-train --nodelist=cn13 --time=4:00:00

MODE=${1:-train}
shift
# remaining args passed to train sbatch call

mkdir -p logs/SLURM

# Get the repo root
# realpath resolves BASH_SOURCE[0] to the absolute path (the script), dirname strips the last element in the path
REPO_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

submit_job() {
    local JOB_ID=$(sbatch --parsable --chdir="$REPO_ROOT" "$@")
    echo "Submitted job: $JOB_ID" >&2
    echo "$JOB_ID"
}

case "$MODE" in
    sync)
        submit_job jobs/sync.job
        ;;
    train)
        submit_job "$@" jobs/train.job
        ;;
    sync-train)
        SYNC_ID=$(submit_job jobs/sync.job)
        TRAIN_ID=$(submit_job --dependency=afterok:$SYNC_ID "$@" jobs/train.job)
        echo "Submitted train job, to start after sync job: $TRAIN_ID"
        ;;
    *)
        echo "Usage: $0 [sync|train|sync-train] [extra sbatch args for train]"
        exit 1
        ;;
esac
