import subprocess


if __name__ == '__main__':
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/unary_closures/dependency -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
                   "-output_dir ./results/hs/unary_closures/pcfg -train_patience 12 "
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax ccg "
                   "-output_dir ./results/hs/unary_closures/ccg -train_patience 12 "
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/no_unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/no_unary_closures/dependency "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/no_unary_closures/pcfg "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda "
    #                "-output_dir ./results/hs/no_unary_closures/ccg "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
