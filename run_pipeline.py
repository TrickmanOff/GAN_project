def main():
    model_name = 'test_model'
    from pipeline.experiment_setup import experiments_storage
    experiments_storage.load_config(model_name)

    from config import run_experiment
    run_experiment.run()


if __name__ == '__main__':
    main()
