TERMNR=2
tmux send-keys -t "$TERMNR" C-c
tmux send-keys -t "$TERMNR" ' \
    rm ../test/out/*.png ; \
    clear && \
    printf "$(tput bold)$(tput setaf 76)==============  Configuring  ==============$(tput sgr0)\n" && \
    cmake .. && \
    printf "$(tput bold)$(tput setaf 76)==============  Building     ==============$(tput sgr0)\n" && \
    make -j4 && \
    printf "$(tput bold)$(tput setaf 76)==============   Running     ==============$(tput sgr0)\n" && \
    ./LabellingTool ../test/out ../test/*.png' Enter
