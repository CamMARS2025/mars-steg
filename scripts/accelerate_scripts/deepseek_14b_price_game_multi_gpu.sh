ACCELERATE_LOG_LEVEL=info accelerate launch --config_file mars_steg/configs/accelerate/multi_gpu.yaml mars_steg/accelerate_training.py \
    experiments/b_sequential_price_game/1_collusion_with_neural_overseer.yaml \
    mars_steg/configs/config_deepseek_14b.yaml \
    scripts/1_testing_with_and_without_penalisation/with_penalise_overwrite.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml   # XXX: 20250212 