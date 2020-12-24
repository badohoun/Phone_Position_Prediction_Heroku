  
mkdir -p -/Phone_Position_Prediction_Heroku/

echo "\
[server]\n\

port = $PORT\n\

enableCORS = false\n\
headless = true\n\
\n\
"  > -/Phone_Position_Prediction_Heroku/config.toml
