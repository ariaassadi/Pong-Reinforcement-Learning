pole_train:
	python3 Pole/train.py --env CartPole-v1

pole_eval:
	python3 Pole/evaluate.py --path Pole/models/CartPole-v1_best.pt

pong_train:
	python3 Pong/train.py ALE/Pong-v5

pong_eval:
	python3 Pong/evaluate.py --path Pong/models/ALE/Pong-v5_best.pt
