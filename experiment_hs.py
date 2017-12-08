import subprocess


if __name__ == '__main__':
    subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm-dropout "
                   "-output_dir ./results/hs/special/bilstm "
                   "-data_dir ./preprocessed/hs", shell=True)

    # with unary closures
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm-dropout "
    #                "-output_dir ./results/hs/unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/unary_closures/dependency "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/unary_closures/pcfg  "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax ccg "
    #                "-output_dir ./results/hs/unary_closures/ccg "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)
    #
    # # without unary
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

    # no thought connection
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/no_thought/no_thought/bilstm "
    #                "-train_patience 15 -no_thought_vector "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/no_unary_closures/no_thought/dependency "
    #                "-train_patience 12 -no_thought_vector "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/no_unary_closures/no_thought/pcfg "
    #                "-train_patience 12 -no_thought_vector "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda "
    #                "-output_dir ./results/hs/no_unary_closures/no_thought/ccg "
    #                "-train_patience 12 -no_thought_vector "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
