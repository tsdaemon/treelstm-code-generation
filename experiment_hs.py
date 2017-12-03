import subprocess


if __name__ == '__main__':
    subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
                   "-output_dir ./results/hs/bilstm "
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
                   "-output_dir ./results/hs/dependency_recursive -train_patience 12 "
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
                   "-output_dir ./results/hs/pcfg_recursive -train_patience 12"
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax ccg "
                   "-output_dir ./results/hs/ccg_recursive -train_patience 12"
                   "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/no_unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/no_unary_closures/dependency_recursive "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/no_unary_closures/pcfg_recursive "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda "
    #                "-output_dir ./results/hs/no_unary_closures/ccg_recursive "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
