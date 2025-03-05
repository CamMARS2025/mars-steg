#/bin/bash

python -m \
    mars_steg.test_memory \
    experiments/b_sequential_price_game/1_collusion_with_neural_overseer.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    scripts/1_testing_with_and_without_penalisation/no_penalise_overwrite.yaml \
    mars_steg/configs/prompt/price_training_deepseek.yaml   # XXX: 20250212
