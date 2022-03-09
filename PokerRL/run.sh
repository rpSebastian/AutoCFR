algo_names="model.2.275"
game_names="subgame3 subgame4"
for algo_name in $algo_names;
do
    for game_name in $game_names;
    do
        tmux_name=${algo_name//./_}_${game_name}
        echo $tmux_name``
        tmux new -s $tmux_name -d
        tmux send-keys -t $tmux_name "cd ~/PokerRL" Enter
        tmux send-keys -t $tmux_name "python -u scripts/run_cfr.py --iters 20000 --algo $algo_name --game $game_name --save" Enter
        # echo $algo_name $game_name
    done
done
