import sys


def main():
    sys.path.insert(0, '/kaggle/input/gan-training')  # for the 'pipeline' to be found
    from config import run_experiment
    run_experiment.run()


if __name__ == '__main__':
    main()
