import concurrent.futures
from tqdm import tqdm


def process_samples_with_executor(
    dataset_to_process,
    workers,
    resume_idx,
    task_handler,
    llm_config,
    save_callback=None,
    save_every=50,
):
    """
    Manages the concurrent execution of tasks using a ThreadPoolExecutor.

    If save_callback is provided, it is called every save_every completions
    with the current completions list so partial results can be persisted.

    On KeyboardInterrupt, pending futures are cancelled immediately and
    partial progress is saved — no hang waiting for in-flight requests.
    """
    completions = []
    futures = []
    saved_count = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        # Submit all tasks to the executor
        for sample in dataset_to_process:
            future = executor.submit(
                task_handler,
                sample,
                llm_config,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            initial=resume_idx,
            desc="Generating completions",
        ):
            result = future.result()
            if result:
                completions.append(result)

            if save_callback and len(completions) - saved_count >= save_every:
                save_callback(completions)
                saved_count = len(completions)

    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving partial progress before exit...")
        if save_callback and completions:
            save_callback(completions)
        executor.shutdown(wait=False, cancel_futures=True)
        return completions

    executor.shutdown(wait=True)
    return completions
