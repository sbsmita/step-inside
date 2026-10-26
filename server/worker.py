"""Step Functions task entry points (run in the same container image on Fargate Graviton).

    python -m server.worker reconstruct [JOB_ID]
    python -m server.worker narrate [JOB_ID]

JOB_ID defaults to the JOB_ID environment variable, which Step Functions sets from the execution input."""
import os
import sys
from pathlib import Path

from .capture import Jobs
from .scenes import SceneIndex
from .storage import make_store


def main(argv: list[str]) -> int:
    if len(argv) == 1 and os.environ.get("JOB_ID"):
        argv = [argv[0], os.environ["JOB_ID"]]
    if len(argv) != 2 or argv[0] not in ("reconstruct", "narrate"):
        print(__doc__)
        return 2
    action, job_id = argv
    store = make_store(Path(os.environ.get("DATA_DIR", "/tmp/data")))
    jobs = Jobs(store, SceneIndex(store))
    job = jobs.get(job_id)
    if job is None:
        print(f"no such job {job_id}")
        return 1
    if action == "reconstruct":
        jobs.run(job, jobs.load_frames(job_id))
        return 0 if job["status"] == "done" else 1
    if job.get("scene_id"):
        jobs.enrich(job["scene_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
