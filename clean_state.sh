#! /bin/bash

for i in $(ls -F data | grep /);
do
    rm -- "data/${i}state.pth" "data/${i}tokens.pkl";
done
