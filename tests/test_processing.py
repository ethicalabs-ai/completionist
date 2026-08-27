import random
import time

from completionist.processing import process_samples_with_executor


def _slow_handler(sample, llm_config):
    # Later samples finish faster, forcing out-of-order completion.
    time.sleep(0.005 * (50 - sample))
    return sample


def test_preserves_submission_order():
    data = list(range(1, 21))
    results = process_samples_with_executor(
        dataset_to_process=data,
        workers=4,
        resume_idx=0,
        task_handler=_slow_handler,
        llm_config={},
    )
    assert results == data


def test_multi_worker_complete_and_ordered():
    rng = random.Random(0)
    data = list(range(1, 101))
    delays = {x: rng.uniform(0.0, 0.02) for x in data}

    def handler(sample, cfg):
        time.sleep(delays[sample])
        return sample

    results = process_samples_with_executor(
        dataset_to_process=data,
        workers=8,
        resume_idx=0,
        task_handler=handler,
        llm_config={},
    )
    assert results == data


def test_save_callback_checkpoints():
    data = list(range(1, 31))
    saved = []

    def callback(completions):
        saved.append(list(completions))

    process_samples_with_executor(
        dataset_to_process=data,
        workers=2,
        resume_idx=0,
        task_handler=lambda x, cfg: x,
        llm_config={},
        save_callback=callback,
        save_every=10,
    )
    assert saved == [list(range(1, 11)), list(range(1, 21)), list(range(1, 31))]


def test_none_results_are_skipped():
    data = ["keep", "skip", "keep"]

    def handler(sample, cfg):
        return None if sample == "skip" else sample

    results = process_samples_with_executor(
        dataset_to_process=data,
        workers=1,
        resume_idx=0,
        task_handler=handler,
        llm_config={},
    )
    assert results == ["keep", "keep"]
