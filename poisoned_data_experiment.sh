python src/train_poisoned.py --model_name logistic_regression --poison_rate 0 --poison_type none
python src/train_poisoned.py --model_name logistic_regression --poison_rate 0.05 --poison_type feature
python src/train_poisoned.py --model_name random_forest --poison_rate 0.10 --poison_type label
python src/train_poisoned.py --poison_type both --poison_rate 0.50
