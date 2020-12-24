  
mkdir -p \Phone_Position_Prediction_Heroku/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
"  > \Phone_Position_Prediction_Heroku/config.toml
