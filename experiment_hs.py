import subprocess


if __name__ == '__main__':
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

    subprocess.run("python main.py -dataset hs -cuda -encoder bi-lstm "
                   "-output_dir ./results/hs/no_thought/no_thought_no_pretrained/bilstm "
                   "-no_thought_vector -no_pretrained_embeds "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax dependency "
                   "-output_dir ./results/hs/no_unary_closures/no_thought_no_pretrained/dependency "
                   "-train_patience 12 "
                   "-no_thought_vector -no_pretrained_embeds "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda -syntax pcfg "
                   "-output_dir ./results/hs/no_unary_closures/no_thought_no_pretrained/pcfg "
                   "-train_patience 12 "
                   "-no_thought_vector -no_pretrained_embeds "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)

    subprocess.run("python main.py -dataset hs -cuda "
                   "-output_dir ./results/hs/no_unary_closures/no_thought_no_pretrained/ccg "
                   "-train_patience 12 "
                   "-no_thought_vector -no_pretrained_embeds "
                   "-data_dir ./preprocessed/hs/no_unary_closures", shell=True)
