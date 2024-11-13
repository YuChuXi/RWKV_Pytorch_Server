#! /bin/bash

for i in $(ls -F data/ | grep /);
do
    /bin/rm -- "data/${i}state.pth" "data/${i}tokens.pkl";
done
