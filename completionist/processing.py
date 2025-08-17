import concurrent.futures
from tqdm import tqdm


def process_samples_with_executor(
    dataset_to_process,
    workers,
    resume_idx,
    task_handler,
    llm_config,
):
    """
    Manages the concurrent execution of tasks using a ThreadPoolExecutor.
    """
    completions = []
    futures = []

    # Using a ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
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

        except KeyboardInterrupt:
            # When interrupted, cancel pending futures and wait for active ones
            print(
                "\nProcess interrupted. Attempting to shut down workers and save progress..."
            )
            executor.shutdown(wait=False, cancel_futures=True)
            print("Executor shutdown initiated.")

        finally:
            # The 'with' statement for the executor automatically calls shutdown
            # at the end of the block, which ensures all resources are cleaned up.
            return completions
