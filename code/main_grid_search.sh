#!/bin/bash

# Define model to be run with first input argument! Options: ANN or XGB
# Endpoint is the seond input argument: OS or PFS
# The rest is set automatically...
model=$1
endpoint=$2

if [[ $model == "ANN" ]]; then
    echo "Running grid search for ANN..."
    
    # Define hyperparameters for grid search
    hidden_size=(10 15 20 25)
    learning_rate=(0.00005 0.0001 0.0005 0.001)
    dropout_prob=(0.0 0.25 0.5)
    batch_size=(32 64)
    weight_decay=(0.0 0.00001)
    
    # Loop over all combinations
    for p1 in "${hidden_size[@]}"; do
        for p2 in "${learning_rate[@]}"; do
            for p3 in "${dropout_prob[@]}"; do
                for p4 in "${batch_size[@]}"; do
                    for p5 in "${weight_decay[@]}"; do
                        command="main_train_ANN.py --hidden_size $p1 --learning_rate $p2 --dropout_prob $p3 --batch_size $p4 --weight_decay $p5 --endpoint $endpoint"
                        echo "Running following command: $command"
                        python $command
                    done
                done
            done        
        done
    done

elif [[ $model == "XGB" ]]; then
    echo "Running grid search for XGB..."
	echo "...not implemented!"

else
    echo "First specified input argument is neither 'ANN' nor 'XGB' or second argument is neither 'PFS' nor 'OS'. Exiting..."
fi

echo "Done!"
