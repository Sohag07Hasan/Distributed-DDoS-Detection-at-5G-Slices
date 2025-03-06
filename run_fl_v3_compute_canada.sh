#!/bin/bash
module load StdEnv/2023
#module load ~/.venvs/fl_env/bin/~/.venvs/fl_env/bin/python/3.11.5
module load r-bundle-bioconductor/3.18
module load gcc/12.3
module load cuda
module load cudnn
module load arrow
module load perl



# List of NUM_FEATURES values to iterate over
NUM_FEATURES_LIST=(27 28 29 30 31 32 33 34 35 36 37 38 39 40 41)

# Loop over NUM_FEATURES values
for NUM_FEATURES in "${NUM_FEATURES_LIST[@]}"; do
  echo "Setting NUM_FEATURES to $NUM_FEATURES..."
  rm -rf ./logs/*.log

  # Update NUM_FEATURES in the config.json file
  jq --argjson num_features "$NUM_FEATURES" '.features.NUM_FEATURES = $num_features' config.json > temp.json && mv temp.json config.json

  # Nested loop for folds
  for FOLD in $(seq 1 5); do
    echo "Running fold $FOLD with NUM_FEATURES=$NUM_FEATURES..."

    # Calculate the port number dynamically (e.g., base port is 8088, increment by 1 for each fold)
    PORT=$((2070 + FOLD + NUM_FEATURES))

    # Check if any process is using the desired port and kill it if found
    echo "Checking if port $PORT is in use..."
    PID=$(lsof -t -i:$PORT)

    if [ ! -z "$PID" ]; then
      echo "Port $PORT is in use by process $PID. Killing the process..."
      kill -9 $PID
    else
      echo "Port $PORT is free."
    fi

    # Update FOLD and SERVER_ADDRESS in the config.json file
    jq --argjson fold "$FOLD" --arg port "0.0.0.0:$PORT" '.dataset_path.FOLD = $fold | .server.SERVER_ADDRESS = $port' config.json > temp.json && mv temp.json config.json

    # Remove any existing server_done.txt file before starting the server
    rm -f server_done.txt

    # Start the server and redirect its output to a log file, capture its PID
    echo "Starting the server for fold $FOLD on port $PORT..."
    START=$(date +%s)
    ~/.venvs/fl_env/bin/python server.py > ./logs/server_fold_${NUM_FEATURES}_$FOLD.log 2>&1 &
    SERVER_PID=$!

    # Give the server some time to start up
    sleep 5

    # Start clients in the background and capture their PIDs
    echo "Starting client 1 for fold $FOLD..."
    ~/.venvs/fl_env/bin/python c1.py > ./logs/client1_fold_${NUM_FEATURES}_$FOLD.log 2>&1 &
    CLIENT1_PID=$!

    echo "Starting client 2 for fold $FOLD..."
    ~/.venvs/fl_env/bin/python c2.py > ./logs/client2_fold_${NUM_FEATURES}_$FOLD.log 2>&1 &
    CLIENT2_PID=$!

    echo "Starting client 3 for fold $FOLD..."
    ~/.venvs/fl_env/bin/python c3.py > ./logs/client3_fold_${NUM_FEATURES}_$FOLD.log 2>&1 &
    CLIENT3_PID=$!

    echo "Starting client 4 for fold $FOLD..."
    ~/.venvs/fl_env/bin/python c4.py > ./logs/client4_fold_${NUM_FEATURES}_$FOLD.log 2>&1 &
    CLIENT4_PID=$!

    # Monitor for server completion by checking for the "done" flag in the file
    echo "Monitoring server completion..."
    while [ ! -f server_done.txt ]; do
      echo "Server is still running..."
      sleep 60  # Check every 60 seconds
    done

    echo "Server process for fold $FOLD has finished."

    # Kill the server process if it's still running
    if ps -p $SERVER_PID > /dev/null; then
      echo "Killing the server process for fold $FOLD..."
      kill -9 $SERVER_PID
    fi

    # Clean up the done flag file
    rm -f server_done.txt

    # End time
    END=$(date +%s)
    DIFF=$((END - START))

    # Display total time
    echo "Total training time for fold $FOLD with NUM_FEATURES=$NUM_FEATURES: $DIFF seconds"
  done
done
