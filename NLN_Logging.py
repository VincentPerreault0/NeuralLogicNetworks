def get_log_files(filename, do_log):
    if do_log and filename != "":
        log_file = open(filename + "_LOG.txt", "w", encoding="utf-8")
        log_files = [log_file]
    else:
        log_files = []
    return log_files


def close_log_files(log_files):
    for log_file in log_files:
        log_file.close()


def print_log(line, verbose, files=[]):
    if verbose:
        print(line)
    for file in files:
        file.write(str(line) + "\n")


def gpu_mem_to_string(gpu_mem):
    if gpu_mem < 1024**3:
        if gpu_mem < 1024**2:
            return str(round(gpu_mem / 1024**1, 2)) + " KB"
        else:
            return str(round(gpu_mem / 1024**2, 2)) + " MB"
    else:
        return str(round(gpu_mem / 1024**3, 2)) + " GB"


def printProgressBar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
