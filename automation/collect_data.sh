#!/bin/bash

for config_file in config/sim_config*.ini; do
  echo "Running simulation with config: $config_file"
  python3 main_loop.py --sim-config "$config_file" --noconfirm
done
