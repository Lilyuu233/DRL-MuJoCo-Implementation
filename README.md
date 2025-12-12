### Procgen env
To try an environment out interactively:
```
python3 -m procgen.interactive --env-name coinrun
```

The keys are: left/right/up/down + q, w, e, a, s, d for the different (environment-dependent) actions. Your score is displayed as "episode_return" in the lower left. At the end of an episode, you can see your final "episode_return" as well as "prev_level_complete" which will be 1 if you successfully completed the level.


### PPO and TRPO
* Run in single process:
```
python3 train_ppo.py --n_proc 1 --env_name dodgeball --eval_mode test # run ppo
python3 train_trpo.py  --n_proc 1 --env_name dodgeball --eval_mode test # run trpo
```

* Run in multiple processes:
```
python3 train_ppo.py --n_proc 4 --env_name dodgeball --eval_mode gen # run ppo
python3 train_trpo.py  --n_proc 4 --env_name dodgeball --eval_mode gen # run trpo
```

* Run in batch tasks:
```
# change tasks.txt
tasks.txt

# kick off train.sh
bash train.sh # in xargs -n 6 -P 4: 6 is the number of columns in tasks, 4 is number of tasks in parallel running
```