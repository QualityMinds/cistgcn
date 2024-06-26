cd ..
mv human-motion-prediction human_motion_prediction 
python3 -m human_motion_prediction.train $1 | tee $2
