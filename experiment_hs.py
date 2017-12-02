import subprocess


if __name__ == '__main__':
    # subprocess.run("main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/bilstm_tanh", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm -no_last_tanh "
    #                "-output_dir ./results/hs/bilstm_no_tanh", shell=True)

    subprocess.run("python main.py -dataset hs -cuda"
                   "-output_dir results/hs/ccg_recursive", shell=True)
