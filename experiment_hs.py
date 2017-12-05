import subprocess


if __name__ == '__main__':
    # with unary closures
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/unary_closures/dependency -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/unary_closures/pcfg -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -syntax ccg "
    #                "-output_dir ./results/hs/unary_closures/ccg -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/unary_closures", shell=True)

    # without unary
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
    #                "-output_dir ./results/hs/no_unary_closures/bilstm "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    # subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
    #                "-output_dir ./results/hs/no_unary_closures/dependency -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
    #                "-output_dir ./results/hs/no_unary_closures/pcfg -train_patience 12 "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
    #
    # subprocess.run("python main.py -dataset hs -cuda "
    #                "-output_dir ./results/hs/no_unary_closures/ccg -train_patience 12 "
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

    # encoder dropout
    # subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm-dropout "
    #                "-output_dir ./results/hs/no_unary_closures/dropout/bilstm "
    #                "-train_patience 12 "
    #                "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
                   "-output_dir ./results/hs/no_unary_closures/dropout/dependency "
                   "-train_patience 12 "
                   # "-encoder_dropout 0.0 "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
                   "-output_dir ./results/hs/no_unary_closures/dropout/pcfg "
                   "-train_patience 12 "
                   # "-encoder_dropout 0.0 "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax ccg "
                   "-output_dir ./results/hs/no_unary_closures/dropout/ccg "
                   "-train_patience 12 "
                   # "-encoder_dropout 0.0 "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
